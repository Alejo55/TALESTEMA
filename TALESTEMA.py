# %% [markdown]
# # Examen Práctico

# %% [markdown]
# #### 3670 COM:01-3900 | Ciencia de datos | 2024 C2

# %% [markdown]
# Alumnos: Martin Lecuona, Rojas Tomas, Alejo Agasi, Stefania Violi, Julian Castellana

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
# ## Como desarrollar el exámen

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

# %%
STUDENTDATAFILE = 'creditos_banco_alumnos.csv'
EVALDATAFILE = 'creditos_banco_evaluacion.csv'
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

# %%
df = pd.read_csv(STUDENTDATAFILE)
df.head()

# %% [markdown]
# # Matriz de Correlación Variables Numéricas

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Seleccionar solo las columnas numéricas
df_numerico = df.select_dtypes(include=['float64', 'int64'])

# Calcular la matriz de correlación
correlacion = df_numerico.corr()

# # Mostrar la matriz de correlación
# print(correlacion)

# Visualizar la matriz de correlación con un mapa de calor
plt.figure(figsize=(8, 6))
sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación de Variables Numéricas')
plt.show()

# %% [markdown]
# Al analizar la matriz, podemos observar que las correlaciones más fuertes están en torno a 0.5 en valor absoluto, representando una correlación moderada. Es por ello que consideramos que no será necesario dropear ninguna de estas variables. (Esto fue debidamente probado y es por ello que se llegó a dicha conclusión))

# %% [markdown]
# # Etapa de preprocesamiento #
# 1. Verificamos valores nulos
# 2. Verificamos valores únicos
# 3. Utilizamos boxplot para encontrar posibles outliers
# 4. Eliminar outliers

# %% [markdown]
# #### Verificamos valores nulos

# %%
#Verificamos valores nulos
print(df.isnull().sum() / len(df) * 100)
# No se encontraron valores nulos en las columnas, por lo tanto no imputamos

# %% [markdown]
# #### Verificamos valores únicos

# %%

# Resumen de estadísticas
print("\nResumen de estadísticas numéricas:")
print(df.describe())
print("\nResumen de estadísticas categóricas:")
print(df.describe(include=['object']))

# %% [markdown]
# #### Analizamos posibles outliers con Boxplot

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
# #### Detectamos outliers, por lo que se los convertirá en nulos para luego imputarlos

# %%
# Seleccionamos columnas numéricas y eliminamos las binarias
numeric_columns = [
    col for col in df.select_dtypes(include=np.number).columns
    if col not in ["FueVeraz", "TuvoEmbargo"]
]

# Mostrar las columnas restantes
print(numeric_columns)


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

# %%
#Verificamos valores nulos
print(df.isnull().sum() / len(df) * 100)

# %%
# Filtrar las filas que tienen al menos un valor nulo
filas_con_nulos = df[df.isnull().any(axis=1)]

# Mostrar las filas que tienen valores nulos
print(filas_con_nulos)

# %% [markdown]
# #### Volvemos a ver el Boxplot

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
# #### Se visualizan los datos con una distribución más equilibrada

# %% [markdown]
# ## Balanceo de variable objetivo "Default"

# %%
print(df['Default'].unique())

# %%
balance_counts = df['Default'].value_counts()

# Mostrar los conteos absolutos
print("Conteos de cada clase:")
print(balance_counts)

balance_proportions = df['Default'].value_counts(normalize=True)
print("\nProporciones de cada clase:")
print(balance_proportions)

# %% [markdown]
# #### Se puede ver que se encuentra balanceada la variable objetivo

# %% [markdown]
# # Split

# %%
from sklearn.model_selection import train_test_split

X = df.drop(['Default'], axis=1)
y = df['Default']

df = X

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=42)

# %% [markdown]
# # Column Transformer (dentro de Pipeline)

# %% [markdown]
# Primero armamos un pipeline para features numericas y otro para features categoricas, que contendran las transformaciones correspondientes como imputaciones y normalizaciones. Luego estos pipelines se agruparan en un column transformer

# %%
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder

numerical_features = df.select_dtypes(include=np.number).columns
categorical_features = df.select_dtypes(exclude=np.number).columns

print(numerical_features)
print("----------")
print(categorical_features)

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
# Ese column transformer se integrara al pipeline definitivo con su correspondiente clasificador

# %% [markdown]
# # PIPELINES KNN Y RANDOM FOREST

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

pipeline_knn = Pipeline(
    steps=[('preprocessor',
            preprocessor), ('classifier', KNeighborsClassifier())])
pipeline_rf = Pipeline(
    steps=[('preprocessor', preprocessor),
           ('classifier',
            RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight="balanced"))])

# %%
from sklearn.model_selection import cross_val_score

print("KNN")
scores_knn = cross_val_score(pipeline_knn,
                             X_train,
                             y_train,
                             cv=5,
                             scoring='accuracy')
print("Puntuaciones en cada fold:", scores_knn)
print("Precisión promedio con validación cruzada:", scores_knn.mean())

# Entrenamos el modelo
pipeline_knn.fit(X_train, y_train)
y_pred_knn = pipeline_knn.predict(X_test)

#############################################

print("RANDOM FOREST")
scores_rf = cross_val_score(pipeline_rf,
                            X_train,
                            y_train,
                            cv=5,
                            scoring='accuracy')
print("Puntuaciones en cada fold:", scores_rf)
print("Precisión promedio con validación cruzada:", scores_rf.mean())

pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)

# %% [markdown]
# # Pipeline Mejor Knn

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

pipeline_knn_best = Pipeline(
    steps=[('preprocessor',
            preprocessor), ('classifier', KNeighborsClassifier())])

param_grid = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11, 15, 20],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
}

# Configuramos GridSearchCV
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
# # Pipeline Mejor Random Forest

# %%
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

pipeline_rf_best = Pipeline(
    steps=[('preprocessor', preprocessor),
           ('classifier',
            RandomForestClassifier(random_state=42, class_weight="balanced"))])

# Definir la cuadrícula de parámetros sin 'log_loss' y sin n_estimators en el clasificador inicial
param_grid_rf = {
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': [3, 4, 5, 6, 7, 8],
    'classifier__min_samples_split': [2, 4, 8, 10, 12, 14],
    'classifier__min_samples_leaf': [2, 4, 6, 8, 10],
    'classifier__max_features': ["sqrt", "log2", None],
    'classifier__n_estimators': [30, 60, 90, 120, 150, 180]
}

rand_search = RandomizedSearchCV(estimator=pipeline_rf_best,
                                 param_distributions=param_grid_rf,
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
# ## Métricas Obtenidas

# %%
from sklearn.metrics import classification_report

print("KNN")
print(classification_report(y_test, y_pred_knn))

print("RANDOM FOREST")
print(classification_report(y_test, y_pred_rf))

print("MEJOR KNN")
print(classification_report(y_test, y_pred_knn_best))

print("MEJOR RANDOM FOREST")
print(classification_report(y_test, y_pred_rf_best))

# %% [markdown]
# # Testing Final

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
