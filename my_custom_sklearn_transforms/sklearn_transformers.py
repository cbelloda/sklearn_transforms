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

from sklearn.preprocessing import RobustScaler

class MRobustScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        from sklearn.preprocessing import RobustScaler 
        data = X.copy()
        rscaler=RobustScaler()
        robustScaler=rscaler.fit(X=data[data.columns.intersection(self.columns)])
        data[data.columns.intersection(self.columns)]=rscaler.transform(data[data.columns.intersection(self.columns)])
        return data