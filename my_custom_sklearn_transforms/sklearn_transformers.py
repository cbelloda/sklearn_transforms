from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class MSimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self

    def transform(self, X): 
        data = X.copy()
        si = SimpleImputer(
            missing_values=np.nan, 
            verbose=0,
            copy=True
        )
        print(X.columns)
        print("\n\nValores nulos en el conjunto de datos ANTES de la transformación SimpleImputer: \n\n{}\n".format(data.isnull().sum(axis = 0)))
        si.fit(X=data[data.columns.intersection(self.columns)])
        data[data.columns.intersection(self.columns)]=pd.DataFrame.from_records(data=si.transform(X=data[data.columns.intersection(self.columns)]),columns=self.columns)
        print("\n\nValores nulos en el conjunto de datos DESPUÉS de la transformación SimpleImputer: \n\n{}\n".format(data.isnull().sum(axis = 0)))
        return data
class MRobustScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X): 
        data = X.copy()
        rscaler=RobustScaler()
        rscaler.fit(X=data[data.columns.intersection(self.columns)])
        data[data.columns.intersection(self.columns)]=rscaler.transform(data[data.columns.intersection(self.columns)])
        return X