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

# %%
print(df.isnull().sum()/len(df)*100)
# no hay  nulos, no se imputa

# verificar unicos
print(df.nunique()/len(df)*100)
# no hay cantidad significativa de verificar unicos en no numericos

# %%
# fig, axs = plt.subplots(1, len(df.columns), figsize=(8, 3))
# for c,i in zip(df.columns, range(len(df.columns))):
#     axs[i].boxplot(df[c])
   
# plt.tight_layout()
# plt.show()

# %%
numeric_columns = df.select_dtypes(include=np.number).columns
for column in numeric_columns:
    plt.figure(figsize=(8, 6))
    plt.boxplot(df[column].dropna())  # Evitar NaN en el boxplot
    plt.title(f'Boxplot de {column}')
    plt.ylabel(column)
    plt.show()

# %%
# importante en "~"
df_numeric = df.select_dtypes(include=np.number)
df_numeric.drop(columns=["FueVeraz"], inplace=True)
df_numeric.drop(columns=["TuvoEmbargo"], inplace=True)

Q1 = df_numeric.quantile(0.25)  # Primer cuartil (Q1)
Q3 = df_numeric.quantile(0.75)  # Tercer cuartil (Q3)
IQR = Q3 - Q1  # Rango intercuartil
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df_numeric[~((df_numeric < lower_bound) | (df_numeric > upper_bound)).any(axis=1)]
print(df.head())

# %%
for column in df_numeric.columns:
    plt.figure(figsize=(8, 6))
    plt.boxplot(df[column].dropna())  # Evitar NaN en el boxplot
    plt.title(f'Boxplot de {column}')
    plt.ylabel(column)
    plt.show()


# %%
# dataumentation
# ver si la columna destino no esta desbalanceada

# %%


# numerical_features = ['Age', 'Fare', 'Parch', 'SibSp']
# categorical_features = ['Pclass', 'Sex', 'Embarked']


# numerical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='mean')),
#     ('scaler', StandardScaler())
# ])

# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore')) 
# ])


