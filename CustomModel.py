import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class WeightedEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = np.array(weights)

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return np.dot(predictions, self.weights)
