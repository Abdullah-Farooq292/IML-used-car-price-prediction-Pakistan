from sklearn.base import BaseEstimator, TransformerMixin


class CarFeatures(BaseEstimator, TransformerMixin):
    """Fill missing mileage/engine with TRAINING medians, then recompute mileage_per_year."""

    def fit(self, X, y=None):
        self.mileage_median_ = X["mileage"].median()
        self.engine_median_ = X["Engine_displacement"].median()
        return self

    def transform(self, X):
        X = X.copy()
        X["mileage"] = X["mileage"].fillna(self.mileage_median_)
        X["Engine_displacement"] = X["Engine_displacement"].fillna(self.engine_median_)
        X["mileage_per_year"] = X["mileage"] / X["car_age"]
        return X
