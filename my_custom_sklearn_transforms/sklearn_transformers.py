from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

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

import sklearn.preprocessing as pre

class MRobustScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X): 
        data = X.copy()
        rscaler=RobustScaler()
        rscaler.fit(X=data[:,self.columns])
        data[:,self.columns]=rscaler.transform(data[:,self.columns])
        return data

