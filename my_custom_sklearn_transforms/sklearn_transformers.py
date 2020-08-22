import sklearn.preprocessing as pre
from sklearn.base import BaseEstimator, TransformerMixin

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
        self.rscaler=pre.RobustScaler()
        
    def fit(self, X, y=None):        
        robustScaler=self.rscaler.fit(X=X[X.columns.intersection(self.columns)])
        return self
    
    def transform(self, X):
        data = X.copy()
        data[data.columns.intersection(self.columns)]=self.rscaler.transform(data[data.columns.intersection(self.columns)])
        return data