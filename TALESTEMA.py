# %% [markdown]
# # Examen Práctico 

# %% [markdown]
# #### 3670 COM:01-3900 | Ciencia de datos | 2024 C2

# %% [markdown]
# Alumnos: Martin Lecuona, Rojas Tomas

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
EVALDATAFILE    = 'creditos_banco_evaluacion.csv'
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
# # Etapa de preprocesamiento #
# 1. Verificamos valores nulos
# 2. Verificamos valores únicos
# 3. Utilizamos boxplot para encontrar posibles outliers
# 4. Eliminar outliers

# %% [markdown]
# #### Verificamos valores nulos

# %%
#Verificamos valores nulos
print(df.isnull().sum()/len(df)*100)
# No se encontraron valores nulos en las columnas, por lo tanto no imputamos

# %% [markdown]
# #### Verificamos valores únicos

# %%

# Resumen de estadísticas
print("\nResumen de estadísticas numéricas:")
print(df.describe())
print("\nResumen de estadísticas categóricas:")
print(df.describe(include=['object']))

#Pasamos a numéricas

# %% [markdown]
# #### Convertimos a numéricos las columnas:
# Tratamientos a realizar en cada columna Label encoding ya que unicamente puede tomar un tipo de valor
# 1. Objetivo
# 2. Default
# 3. EsPropietario

# %%
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder


# # Inicializamos los codificadores
# label_encoder_objetivo = LabelEncoder()
# label_encoder_default = LabelEncoder()
# label_encoder_propietario = LabelEncoder()

# # Codificamos y actualizamos la columna 'Objetivo'
# df['Objetivo'] = label_encoder_objetivo.fit_transform(df['Objetivo'])

# # Codificamos y actualizamos la columna 'default'
# df['Default'] = label_encoder_default.fit_transform(df['Default'])

# # Codificamos y actualizamos la columna 'esPropietario'
# df['esPropietario'] = label_encoder_propietario.fit_transform(df['esPropietario'])

# print(df[['Objetivo', 'Default', 'esPropietario']].head())



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
# #### Detectamos outliers, por lo que intentamos limpiarlos usando IQR

# %%
#Seleccionamos columnas numéricas
df_numeric = df.select_dtypes(include=np.number)

#Elimanos columnas "binarias"

df_numeric.drop(columns=["FueVeraz"], inplace=True)
df_numeric.drop(columns=["TuvoEmbargo"], inplace=True)

#Primer cuartil (Q1)
Q1 = df_numeric.quantile(0.25)  

#Tercer cuartil (Q3)
Q3 = df_numeric.quantile(0.75)  

#Rango intercuartil
IQR = Q3 - Q1  

#Definimos los límites inferior y superior
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

#Filtramos los datos y actualizamos el dataFrame. "~ Lo utilizamos para tomar los datos que deberiamos eliminar"
mask = ~((df_numeric < lower_bound) | (df_numeric > upper_bound)).any(axis=1)

# Filtrar el DataFrame original usando la máscara y asignarlo de nuevo a df
df = df[mask].reset_index(drop=True)

#Verificamos que se hayan eliminado los outliers
print(df.head())

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
# 
# #### Realizamos una matriz de correlación para comprender como se relacionan las columnas de df

# %%
# Calcular la matriz de correlación
# correlation_matrix = df.corr()

# correlation_matrix = df.corr()

# # Visualizar la matriz de correlación
# plt.figure(figsize=(10, 8))
# plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
# plt.colorbar()
# plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
# plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
# plt.title("Matriz de Correlación")
# plt.tight_layout()
# plt.show()



# %%
# df['Default'] = df['Default'].replace({'paid off': 0, 'default': 1})

# %%


from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

X = df.drop(['Default'], axis=1)
y = df['Default']

df = X

numerical_features = df.select_dtypes(include=np.number).columns
categorical_features = df.select_dtypes(exclude=np.number).columns

print(numerical_features)
print("----------")
print(categorical_features)


# numerical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='mean')),
#     ('scaler', StandardScaler())
# ])
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')) 
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
pipeline_knn = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42,class_weight="balanced"))
])


# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
from sklearn.metrics import classification_report
# Entrenamos el modelo
pipeline_knn.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)

# Evaluación
from sklearn.metrics import accuracy_score

y_pred_knn = pipeline_knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
y_pred_rf = pipeline_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# %% [markdown]
# # Mejor Knn

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


param_grid = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11, 15, 20], 
    'classifier__weights': ['uniform', 'distance'],        
    'classifier__metric': ['euclidean', 'manhattan', 'minkowski']  
}

pipeline_knn_best = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])


# Configuramos GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline_knn,
    param_grid=param_grid,
    cv=5,               # Validación cruzada con 5 folds
    scoring='accuracy',  # Métrica de evaluación
    n_jobs=-1            # Usar todos los núcleos disponibles
)

grid_search.fit(X_train, y_train)

print("Mejores hiperparámetros:", grid_search.best_params_)

y_pred_optimized = grid_search.predict(X_test)
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)

# %%
print("KNN")
print(classification_report(y_test, y_pred_knn))
print("RANDOM FOREST")
print(classification_report(y_test, y_pred_rf))
print("MEJOR KNN")
print(classification_report(y_test, y_pred_optimized))

# %%
EVALUACION = False
best_clf = pipeline_rf #Asignar aqui el mejor clasificador posible (previamente entrenado)
#best_clf = pl
#Leemos el dataset de evaluación, simulando producción
if EVALUACION==False:
    df = pd.read_csv(STUDENTDATAFILE)
    _, df = train_test_split(df, test_size=0.3, random_state=42)
else:
    df = pd.read_csv(EVALDATAFILE)
#Dividimos en target y predictoras

X_Eval = df.drop("Default", axis=1)
y_Eval = df["Default"]

#Evaluación final

y_pred = best_clf.predict(X_Eval) # esto debe ser un pipeline completo
print(classification_report(y_Eval, y_pred))


