from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import numpy as np

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

class MRobustScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, XX): 
        data = XX.copy()
        rscaler=RobustScaler()
        rscaler.fit(X=XX[XX.columns.intersection(self.columns)])
        XX[XX.columns.intersection(self.columns)]=rscaler.transform(XX[XX.columns.intersection(self.columns)])
        return XX


class MSimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self

    def transform(self, XX): 
        data = XX.copy()
        si = SimpleImputer(
            missing_values=np.nan, 
            verbose=0,
            copy=True
        )
        si.fit(X=XX[XX.columns.intersection(self.columns)])
        XX[XX.columns.intersection(self.columns)]=pd.DataFrame.from_records(data=si.transform(X=XX[XX.columns.intersection(self.columns)]),columns=self.columns)
        return XX

