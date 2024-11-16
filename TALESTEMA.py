# %% [markdown]
# # Examen Práctico

# %% [markdown]
# #### 3670 COM:01-3900 | Ciencia de datos | 2024 C2

# %% [markdown]
# Alumnos: Martin Lecuona, Tomas Rojas, Alejo Agasi, Stefania Violi, Julian Castellana

# %% [markdown]
# ## Enunciado

# %% [markdown]
# Se tiene un dataset con datos del historial de solicitantes a quienes se le otorgaron créditos y su situación final como deudores o pagadores. La entidad tiene que determinar a quienes entregar o no un crédito en función de su propensión a caer en "default". Desarrolle un proceso que clasifique deudores y pagadores. Observe que la clase de interés es "default", y debido a nuevas políticas de encaje bancario se ha expresado el objetivo de evitar tanto como sea posible entregar créditos a deudores (a costa naturalmente de perder algún posible crédito a pagadores). Maximice la métrica correspondiente sin modificar threshold.

# %% [markdown]
#  Las columnas tienen nombres descriptivos, pero para mas información:

# %% [markdown]
# Importe: Cuando dinero esta pidiendo prestado</BR>
# añosPago: Tiempo para pagar el crédito</BR>
# IngresoAnuales: Ingresos anuales del solicitante</BR>
# RelacionIngresoDeuda: Ratio entre sus ingresos y la deuda</BR>
# RelacionCuotaDeuda: Ratio entre sus ingresos y la cuota</BR>
# PendienteEnTarjeta: Pendiente de pago en tarjetas de crédito</BR>
# UsoCreditoTarjeta: Volumen de dinero que maneja con sus instrumentos de crédito</BR>
# Objetivo: ¿Para que quiere el préstamos?</BR>
# esPropietario: ¿Es propietario del su casa?</BR>
# FueVeraz: ¿Alguna vez estuvo en el veraz?</BR>
# TuvoEmbargo: ¿Tuvo algun embargo o situación judicial?</BR>
# Cuentas: Cantidad de cuentas que maneja</BR>
# PuntuacionGeneral: Puntuación crediticia otorgada por un organismo regular</BR>
# Default: Si pagó o no el crédito</BR>
# AntiguedadLaboral: Antiguedad laboral</BR>

# %% [markdown]
# ## Como desarrollar el examen

# %% [markdown]
# A partir del dataset realice todas las acciones para poder llegar al mejor modelo, explique brevemente en los fundamentos de sus transformaciones o acciones en general.

# %% [markdown]
# La nota derivará de: </BR>
# 1.La calidad de la clasificación realizada</BR>
# 2.La fundamentación de los pasos realizados</BR>
# 3.Lo sencillo de llevar a producción el desarrollo</BR>
#
#

# %% [markdown]
# Los docentes evaluaran su clasificador utilizando un conjunto de datos del dataset "fuera de la caja" (out of the box, al que usted no tiene acceso). Para minimizar la posible diferencia entre su medición y la medición del docente, recuerde y aplique conceptos de test, validación cruzada y evite los errores comunes de sesgo de selección y fuga de datos. Ej: "10. Common pitfalls and recommended practices" disponible en "https://scikit-learn.org/stable/common_pitfalls.html"

# %% [markdown]
# Al final del notebook encontrará un bloque de código que lee la muestra adicional (a la que usted no tiene acceso) si EVALUACION==True, en caso contrario solo lee una submuestra del conjunto original para validar que el código funciona. Desarrolle el notebook como considere, para finalmente asignar el mejor clasificador que usted haya obtenido remplazando en f_clf = None, None por su clasificador. Implemente todas las transformaciones entre esa línea y la predición final (Evitando al fuga de datos).Puede dejar funcionando implementaciones alternativas que no prosperaron en notebooks separados. En cuanto comience con el desarrollo informe a los docentes el nombre del repositorio.
#

# %% [markdown]
# # 1. Importación de Bibliotecas

# %%
STUDENTDATAFILE = 'creditos_banco_alumnos.csv'
EVALDATAFILE = 'creditos_banco_evaluacion.csv'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# %% [markdown]
# # 2. Carga del Dataset
#
# Observamos la estructura del dataset, las columnas y los tipos de datos

# %%
df = pd.read_csv(STUDENTDATAFILE)
df.head()

# %% [markdown]
# # 3. Análisis Exploratorio de Datos

# %% [markdown]
# ## 3.1 Análisis de Correlación entre Variables Numéricas
#
# En esta etapa, se calcula y visualiza la matriz de correlación entre las variables numéricas del conjunto de datos. El objetivo es identificar las relaciones lineales entre las características, lo que puede ayudar a detectar posibles redundancias (variables altamente correlacionadas) y a comprender mejor las interacciones entre las variables.

# %%
# Seleccionar solo las columnas numéricas
df_numerico = df.select_dtypes(include=['float64', 'int64'])

# Calcular la matriz de correlación
correlacion = df_numerico.corr()

# Visualizar la matriz de correlación con un mapa de calor
plt.figure(figsize=(8, 6))
sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación de Variables Numéricas')
plt.show()

# %% [markdown]
# Al analizar la matriz, podemos observar que las correlaciones más fuertes están en torno a 0.5 en valor absoluto, representando una correlación moderada. Es por ello que consideramos que no será necesario dropear ninguna de estas variables. (Esto fue debidamente probado y es por ello que se llegó a dicha conclusión)

# %% [markdown]
# ## 3.2 Resumen de Estadísticas
#
# Esto nos permite tener un panorama general del dataset, identificar patrones iniciales, y detectar problemas potenciales antes de avanzar al modelado o análisis más profundo.

# %%
# Resumen de estadísticas
print("\nResumen de estadísticas numéricas:")
df.describe().T

# %%
print("\nResumen de estadísticas categóricas:")
df.describe(include=['object'])

# %% [markdown]
# ## 3.3 Valores nulos
#
# Dado que ninguna de las columnas presentó valores nulos, no será necesario aplicar técnicas de imputación (a priori).

# %%
df.isnull().sum() / len(df) * 100

# %% [markdown]
# ## 3.4 Outliers

# %% [markdown]
# ### 3.4.1 Análisis de Outliers con Boxplot
#
# Realizamos un análisis de los outliers (valores atípicos) en las variables numéricas del conjunto de datos utilizando boxplots.
# Cada boxplot muestra la distribución de datos en cada variable y destaca los posibles valores atípicos (outliers) mediante puntos que se ubican fuera del rango intercuartílico.

# %%
#Obtenemos únicamente las columnas que sean del tipo numéricas
numeric_columns = df.select_dtypes(include=np.number).columns

# Número de columnas numéricas
n = len(numeric_columns)

# Ajustar el tamaño
plt.figure(figsize=(4 * n, 6))

for i, column in enumerate(numeric_columns):
    plt.subplot(1, n, i + 1)
    plt.boxplot(df[column].dropna())  # Evitar NaN en el boxplot
    plt.title(f'Boxplot de {column}')
    plt.ylabel(column)

plt.tight_layout()
plt.show()

# %% [markdown]
# Descargar o expandir la anterior imagen para una mejor visualización.
#
# Dado que se detectaron outliers en el análisis previo de boxplots, se decidió en los próximos pasos convertir estos valores atípicos en valores nulos (NaN) para su posterior imputación. Esta técnica permite que los datos extremos no afecten el análisis y modelado, pero mantiene el tamaño de las muestras para poder tratarlos en pasos posteriores.

# %% [markdown]
# ### 3.4.2 Selección de columnas numéricas (no binarias)
#
# Se eliminan las columnas numéricas binarias ya que los cálculos de estadísticas como cuartiles, rango intercuartil, y outliers no tienen sentido en este contexto.

# %%
# Seleccionamos columnas numéricas y eliminamos las binarias
numeric_columns = [
    col for col in df.select_dtypes(include=np.number).columns
    if col not in ["FueVeraz", "TuvoEmbargo"]
]

# Mostrar las columnas restantes
numeric_columns

# %% [markdown]
# ### 3.4.3 Identificación y Reemplazo de Outliers usando el Rango Intercuartílico
#
# Para manejar los valores atípicos en las variables numéricas, utilizamos el rango intercuartílico (IQR) para identificarlos y marcarlos como nulos (NaN).
# Calculamos los límites inferior y superior basados en el IQR y reemplazamos los valores fuera de estos límites con NaN. Esto permite identificar y tratar los outliers sin eliminarlos del conjunto de datos, facilitando su imputación en etapas posteriores.


# %%
# Función para identificar outliers usando el rango intercuartílico (IQR)
def reemplazar_outliers_por_nulos(df, columnas):
    for columna in columnas:
        # Calcula el IQR (Q3 - Q1)
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        # Reemplaza los outliers por NaN
        df[columna] = df[columna].apply(lambda x: np.nan if (
            x < limite_inferior or x > limite_superior) else x)


# Reemplazar outliers en las columnas especificadas
reemplazar_outliers_por_nulos(df, numeric_columns)

# %% [markdown]
# ### 3.4.4 Verificación del Porcentaje de Valores Nulos
#
# Luego de reemplazar los outliers por valores nulos, calculamos el porcentaje de datos faltantes en cada columna para tener una visión clara de cuántos valores se deben imputar. Este análisis es fundamental para planificar el tratamiento de valores nulos en las siguientes etapas y garantizar la integridad del conjunto de datos.

# %%
df.isnull().sum() / len(df) * 100

# %% [markdown]
# ### 3.4.5 Visualización de Filas con Valores Nulos
#
# Después de identificar el porcentaje de valores nulos en cada columna, filtramos y visualizamos las filas que contienen al menos un valor nulo. Este paso nos permite observar directamente cuáles registros tienen datos faltantes en el conjunto de datos, ayudándonos a analizar patrones de datos ausentes y a planificar la estrategia de imputación adecuada. En este caso, observamos un total de 4,198 filas con valores nulos en distintas columnas.

# %%
# Filtrar las filas que tienen al menos un valor nulo
filas_con_nulos = df[df.isnull().any(axis=1)]

# Mostrar las filas que tienen valores nulos
filas_con_nulos

# %% [markdown]
# ### 3.4.6 Nueva visualización de los Boxplots
#
# En esta etapa, realizamos una nueva visualización de los boxplots para todas las variables numéricas después de haber reemplazado los outliers con valores nulos. Esto nos permite verificar si los valores extremos han sido eliminados, facilitando así una evaluación más precisa de la distribución de los datos y asegurando que las variables estén preparadas para el siguiente paso en el análisis.
# Como resultado, se visualizan los datos con una distribución más equilibrada, lo que asegura que las variables estén listas para el siguiente paso en el análisis, optimizando la calidad y la consistencia de los datos para su posterior procesamiento.

# %%
#Obtenemos únicamente las columnas que sean del tipo numéricas
numeric_columns = df.select_dtypes(include=np.number).columns

# Número de columnas numéricas
n = len(numeric_columns)

# Ajustar el tamaño
plt.figure(figsize=(4 * n, 6))

for i, column in enumerate(numeric_columns):
    plt.subplot(1, n, i + 1)
    plt.boxplot(df[column].dropna())  # Evitar NaN en el boxplot
    plt.title(f'Boxplot de {column}')
    plt.ylabel(column)

plt.tight_layout()
plt.show()

# %% [markdown]
# Descargar o expandir la anterior imagen para una mejor visualización y comparación con el primer Boxplot.

# %% [markdown]
# ## 3.5 Distribución de la variable objetivo "Default"
#
# En esta sección, analizamos el balance de la variable objetivo "Default", que representa si un cliente ha pagado correctamente ('paid off') o ha incurrido en un incumplimiento ('default'). Primero, calculamos los conteos absolutos de cada clase para conocer cuántos registros pertenecen a cada categoría. Luego, calculamos las proporciones relativas de cada clase para entender la distribución de las clases en relación con el total de datos.
#
# Este análisis es crucial para determinar si existe un desbalance significativo entre las clases, lo que podría afectar la capacidad del modelo para predecir correctamente. Si se encuentra un desbalance, se podrían aplicar técnicas de balanceo como oversampling o undersampling para equilibrar las clases y evitar que el modelo esté sesgado hacia la clase mayoritaria.

# %%
print(df['Default'].unique())

# %%
balance_counts = df['Default'].value_counts()
print("Conteos de cada clase:")
print(balance_counts)

balance_proportions = df['Default'].value_counts(normalize=True)
print("\nProporciones de cada clase:")
print(balance_proportions)

# %% [markdown]
# Se puede ver que la variable objetivo está balanceada, ya que ambas clases tienen la misma cantidad de registros y proporciones. Esto asegura que el modelo no se verá sesgado hacia ninguna de las clases, por lo que no es necesario aplicar técnicas de balanceo.

# %% [markdown]
# # 4. Preparación de Datos

# %% [markdown]
# En esta etapa, el conjunto de datos se divide en dos partes: una para entrenar el modelo y otra para evaluarlo. Para ello, utilizamos la función train_test_split de scikit-learn, que separa los datos en un conjunto de entrenamiento y un conjunto de prueba.
#
# Esta división es fundamental, ya que permite entrenar el modelo con un conjunto de datos y evaluarlo en un conjunto independiente. De esta manera, se minimiza el riesgo de sobreajuste y se obtiene una evaluación más realista del rendimiento del modelo.

# %%
X = df.drop(['Default'], axis=1)
y = df['Default']

df = X

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=42)

# %% [markdown]
# # 5. Preprocesamiento de Datos
#
# En esta etapa, se crean pipelines separados para las características numéricas y categóricas, con transformaciones específicas para cada tipo de dato.
# Para las características numéricas, se realiza imputación de valores faltantes utilizando la media y normalización con StandardScaler.
# Para las características categóricas, se realiza imputación utilizando el valor más frecuente y codificación mediante OneHotEncoder para convertir las categorías en variables binarias, ignorando las categorías no vistas durante el entrenamiento.
#
# Finalmente, ambos pipelines se combinan en un ColumnTransformer, que aplica las transformaciones correspondientes a cada tipo de característica en paralelo, asegurando que los datos sean procesados correctamente antes de alimentar el modelo.

# %%
numerical_features = df.select_dtypes(include=np.number).columns
categorical_features = df.select_dtypes(exclude=np.number).columns

print(f"Numerical Features:\n{numerical_features}")
print(f"\nCategorical Features:\n{categorical_features}")

numerical_transformer = Pipeline(
    steps=[('imputer',
            SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])

categorical_transformer = Pipeline(
    steps=[('imputer', SimpleImputer(strategy='most_frequent')
            ), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[(
    'num', numerical_transformer,
    numerical_features), ('cat', categorical_transformer,
                          categorical_features)])

# %% [markdown]
# Este ColumnTransformer formará parte del pipeline definitivo, que incluirá el clasificador correspondiente, integrando así las transformaciones de datos y el modelo de clasificación en un único flujo de trabajo automatizado. Esto optimiza el proceso de entrenamiento y evaluación.

# %% [markdown]
# # 6. Definición de Modelos
#
# En esta etapa, se crean dos pipelines para entrenar y evaluar modelos de clasificación utilizando K-Nearest Neighbors (KNN) y Random Forest. Ambos pipelines integran las transformaciones de preprocesamiento definidas anteriormente, utilizando el ColumnTransformer para asegurar que los datos numéricos y categóricos sean procesados adecuadamente antes de alimentar al clasificador.

# %%
pipeline_knn = Pipeline(
    steps=[('preprocessor',
            preprocessor), ('classifier', KNeighborsClassifier())])

pipeline_rf = Pipeline(
    steps=[('preprocessor', preprocessor),
           ('classifier',
            RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight="balanced"))])

# %% [markdown]
# # 7. Evaluación de Modelos

# %% [markdown]
# En esta etapa, evaluamos el rendimiento de los modelos de clasificación K-Nearest Neighbors (KNN) y Random Forest utilizando validación cruzada con 5 particiones (folds).
#
# La validación cruzada permite evaluar el rendimiento de los modelos de manera más robusta, al entrenarlos y validarlos múltiples veces en diferentes subconjuntos de los datos de entrenamiento, reduciendo el riesgo de sobreajuste.
#
# Realizar la validación cruzada antes del ajuste completo del modelo tiene dos beneficios principales:
#
# 1- Medir la capacidad general de aprendizaje del modelo: Permite estimar cómo podría generalizar el modelo a nuevos datos sin haber visto el conjunto de prueba, evitando un sesgo en las métricas de evaluación.
#
# 2- Evitar el sobreajuste en la evaluación: Al realizar el ajuste completo (fit) después de la validación cruzada, se garantiza que las métricas de validación no estén influenciadas por un modelo entrenado con todo el conjunto de entrenamiento.
#
# Una vez realizada la validación cruzada, se entrena cada modelo con el conjunto de entrenamiento completo y se evalúa su capacidad de generalización utilizando el conjunto de prueba.

# %%
print("KNN")
scores_knn = cross_val_score(pipeline_knn,
                             X_train,
                             y_train,
                             cv=5,
                             scoring='accuracy')
print("Puntuaciones en cada fold:", scores_knn)
print("Accuracy promedio con validación cruzada:", scores_knn.mean())
pipeline_knn.fit(X_train, y_train)
y_pred_knn = pipeline_knn.predict(X_test)

print("\nRANDOM FOREST")
scores_rf = cross_val_score(pipeline_rf,
                            X_train,
                            y_train,
                            cv=5,
                            scoring='accuracy')
print("Puntuaciones en cada fold:", scores_rf)
print("Accuracy promedio con validación cruzada:", scores_rf.mean())
pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)

# %% [markdown]
# Los resultados indican que el modelo Random Forest supera al modelo KNN en términos de Accuracy promedio, lo que sugiere que Random Forest podría ser más adecuado para este conjunto de datos.

# %% [markdown]
# # 8. Optimización de Modelos con Búsqueda de Hiperparámetros

# %% [markdown]
# ## 8.1 Optimización del Modelo KNN
#
# En esta etapa, se realiza la búsqueda de hiperparámetros óptimos para el modelo K-Nearest Neighbors (KNN) utilizando GridSearchCV. Este procedimiento evalúa una combinación de valores para los hiperparámetros más relevantes del modelo, como el número de vecinos (n_neighbors), el tipo de ponderación de los vecinos (weights), y la métrica de distancia utilizada (metric). La búsqueda se realiza a través de validación cruzada con 5 particiones para garantizar una evaluación robusta del rendimiento.
#
# Con este proceso, el mejor modelo KNN se obtiene a partir de la combinación óptima de hiperparámetros, lo que maximiza la accuracy en el conjunto de datos de entrenamiento.
#
# Una vez identificado el mejor conjunto de parámetros, el modelo resultante se entrena completamente con los datos de entrenamiento (X_train, y_train) y se evalúa utilizando el conjunto de prueba (X_test), permitiendo medir su capacidad de generalización.

# %%
pipeline_knn_best = Pipeline(
    steps=[('preprocessor',
            preprocessor), ('classifier', KNeighborsClassifier())])

param_grid = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11, 15, 20],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
}

grid_search = GridSearchCV(
    estimator=pipeline_knn_best,
    param_grid=param_grid,
    cv=5,  # Validación cruzada con 5 folds
    scoring='accuracy',  # Métrica de evaluación
    n_jobs=-1  # Usar todos los núcleos disponibles
)

grid_search.fit(X_train, y_train)
y_pred_knn_best = grid_search.predict(X_test)
pipeline_knn_best = grid_search.best_estimator_

print("\nKNN - Mejor Estimador:")
pipeline_knn_best

# %% [markdown]
# ## 8.2 Optimización del Modelo Random Forest
#
# En esta etapa, se utiliza RandomizedSearchCV para ajustar los hiperparámetros del clasificador Random Forest. Este enfoque permite explorar un espacio más amplio de hiperparámetros en comparación con GridSearchCV, ya que, en lugar de probar todas las combinaciones posibles, selecciona aleatoriamente un conjunto de combinaciones para probar, lo que puede ser más eficiente.
#
# Los hiperparámetros ajustados incluyen el número de árboles en el bosque (n_estimators), la profundidad máxima de los árboles (max_depth), el criterio de división de los nodos (criterion), el número mínimo de muestras para dividir un nodo (min_samples_split), entre otros. El objetivo de este proceso es identificar la mejor combinación de estos parámetros para maximizar la accuracy del modelo y asegurar su rendimiento óptimo.
#
# Una vez identificados los mejores hiperparámetros mediante RandomizedSearchCV, el modelo resultante se entrena completamente con los datos de entrenamiento (X_train, y_train). Este modelo optimizado es posteriormente evaluado utilizando el conjunto de prueba (X_test), permitiendo medir su capacidad de generalización.

# %%
pipeline_rf_best = Pipeline(
    steps=[('preprocessor', preprocessor),
           ('classifier',
            RandomForestClassifier(random_state=42, class_weight="balanced"))])

param_distributions = {
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': [3, 4, 5, 6, 7, 8],
    'classifier__min_samples_split': [2, 4, 8, 10, 12, 14],
    'classifier__min_samples_leaf': [2, 4, 6, 8, 10],
    'classifier__max_features': ["sqrt", "log2", None],
    'classifier__n_estimators': [30, 60, 90, 120, 150, 180]
}

rand_search = RandomizedSearchCV(estimator=pipeline_rf_best,
                                 param_distributions=param_distributions,
                                 n_iter=50,
                                 cv=5,
                                 scoring='accuracy',
                                 random_state=42,
                                 n_jobs=-1)

rand_search.fit(X_train, y_train)
y_pred_rf_best = rand_search.predict(X_test)
pipeline_rf_best = rand_search.best_estimator_

print("\nRandom Forest - Mejor Estimador:")
pipeline_rf_best

# %% [markdown]
# # 9. Evaluación Final de Modelos
#
# En esta etapa, evaluamos el rendimiento de los modelos utilizando varias métricas de clasificación para comprender su desempeño en los datos de prueba. Se utilizan métricas como precisión, recall, f1-score y soporte para cada clase, lo cual nos proporciona una visión completa de cómo el modelo está clasificando las instancias de las clases 'paid off' y 'default'.

# %%
print("KNN")
print(classification_report(y_test, y_pred_knn))

print("\nRANDOM FOREST")
print(classification_report(y_test, y_pred_rf))

print("\nMEJOR KNN")
print(classification_report(y_test, y_pred_knn_best))

print("\nMEJOR RANDOM FOREST")
print(classification_report(y_test, y_pred_rf_best))

# %% [markdown]
# Después de evaluar múltiples modelos de clasificación, el Mejor Random Forest (Random Forest Optimizado) fue seleccionado por su buen rendimiento en accuracy (64%), su equilibrio entre precision y recall para ambas clases, y su capacidad de generalización frente a nuevos datos. Además, este modelo se ajusta mejor a los requerimientos del problema al ofrecer una clasificación confiable para ambas clases, minimizando los errores críticos asociados con la predicción de default.

# %% [markdown]
# # 10. Evaluación en Simulación de Producción
#
# En esta etapa, simulamos un escenario de producción donde el modelo seleccionado (Random Forest Optimizado) se evalúa sobre un conjunto de datos de evaluación independiente, que no ha sido utilizado durante el proceso de entrenamiento. El objetivo es verificar cómo el modelo generaliza a datos no vistos y obtener las métricas finales de desempeño.

# %%
EVALUACION = False
best_clf = pipeline_rf_best  #Asignar aqui el mejor clasificador posible (previamente entrenado)

#Leemos el dataset de evaluación, simulando producción
if EVALUACION == False:
    df = pd.read_csv(STUDENTDATAFILE)
    _, df = train_test_split(df, test_size=0.3, random_state=42)
else:
    df = pd.read_csv(EVALDATAFILE)
#Dividimos en target y predictoras

X_Eval = df.drop("Default", axis=1)
y_Eval = df["Default"]

#Evaluación final
y_pred = best_clf.predict(X_Eval)  # esto debe ser un pipeline completo
print(classification_report(y_Eval, y_pred))

# %% [markdown]
# # 11. Conclusión Final
#
# El modelo Random Forest Optimizado ha mostrado un rendimiento robusto en la clasificación de las categorías "default" y "paid off". Con una accuracy general de 0.67, se observa que tanto la precisión como el recall están equilibrados entre ambas clases, lo que indica una clasificación efectiva sin sesgo hacia ninguna de las categorías. En particular, el precision y recall de la clase "default" son ligeramente superiores a los de la clase "paid off", lo que puede reflejar una ligera preferencia del modelo por predecir esta clase con mayor precisión.
#
# A nivel general, el modelo muestra una f1-score promedio de 0.67, lo que sugiere que el modelo es capaz de equilibrar bien la precisión y el recall, minimizando tanto los falsos positivos como los falsos negativos. La optimización de hiperparámetros, como el número de árboles en el bosque y la profundidad máxima de los árboles, ha contribuido a mejorar la capacidad predictiva del Random Forest, lo que lo convierte en un modelo adecuado para esta tarea de clasificación.
#
# En resumen, el Random Forest optimizado es una opción sólida para la clasificación en este contexto, logrando un buen balance entre las métricas de precisión y recall, y brindando una capacidad de generalización adecuada en el conjunto de prueba.
