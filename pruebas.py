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

df = pd.read_csv(STUDENTDATAFILE)


# verificar nulos
print(df.isnull().sum()/len(df)*100)
# no hay  nulos, no se imputa

# verificar unicos
print(df.nunique()/len(df)*100)
# no hay cantidad significativa de verificar unicos en no numericos

# boxplot all together
numeric_columns = df.select_dtypes(include=np.number).columns
# plt.boxplot(df[numeric_columns].values)
# plt.xticks(range(1, len(numeric_columns) + 1), numeric_columns, rotation=90)
# plt.title('Boxplot de columnas numéricas')
# plt.show()
# boxplot indv

# for column in numeric_columns:
#     plt.figure(figsize=(8, 6))
#     plt.boxplot(df[column].dropna())  # Evitar NaN en el boxplot
#     plt.title(f'Boxplot de {column}')
#     plt.ylabel(column)
#     plt.show()
# muchos outliers
# Importe,añosPago,IngresoAnuales,RelacionIngresoDeuda,RelacionCuotaDeuda,PendienteEnTarjeta,UsoCreditoTarjeta,Objetivo,esPropietario,FueVeraz,TuvoEmbargo,Cuentas,PuntuacionGeneral,Default,AntiguedadLaboral
numeric_columns = df.select_dtypes(include=np.number).columns
numeric_columns.drop("FueVeraz")
# borrar los datos outliers
df_clean = df.copy()

for column in numeric_columns:
    Q1 = df[column].quantile(0.25)  # Primer cuartil
    Q3 = df[column].quantile(0.75)  # Tercer cuartil
    IQR = Q3 - Q1  # Rango intercuartílico
    lower_bound = Q1 - 1.5 * IQR  # Limite inferior
    upper_bound = Q3 + 1.5 * IQR  # Limite superior
    # Filtrar solo los valores dentro de los límites, pero no modificar otras columnas
    df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]


print(df_clean.head())
numeric_columns = df_clean.select_dtypes(include=np.number).columns
#muestra mejora de outliers
for column in numeric_columns:
    plt.figure(figsize=(8, 6))
    plt.boxplot(df_clean[column].dropna())  # Evitar NaN en el boxplot
    plt.title(f'Boxplot de {column}')
    plt.ylabel(column)
    plt.show()

exit(-1)


# Atencion: Esto es un ejemplo solo para verificar que la plantilla funciona, remplazar.
X = df.drop(columns='Default')
y = df['Default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
preprocessor = ColumnTransformer(
    transformers=[
        ('SimpleImputacion', SimpleImputer(), ['añosPago', 'IngresoAnuales']),
    ])
#Pipeline ejemplo
pl = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=20))
])
pl.fit(X_train, y_train)
EVALUACION = False
# best_clf = None #Asignar aqui el mejor clasificador posible (previamente entrenado)
best_clf = pl
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