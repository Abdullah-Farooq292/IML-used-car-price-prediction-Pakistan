"""Predict the asking price (Rs) of a used car in Pakistan.

    from predict import predict_price
    predict_price("Toyota", "Corolla", "Lahore", year=2018, mileage=60000, engine_displacement=1300)

Use make/model/city spelled as in the dataset (lists are in model/metadata.json).
Unknown makes, models or cities fall back to the average price.
"""
import os

import joblib
import numpy as np
import pandas as pd

from car_features import CarFeatures  # noqa: F401  (needed so joblib can rebuild the pipeline)

REFERENCE_YEAR = 2024  # car_age was computed as 2024 - year when the model was trained
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "price_model.joblib")
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(_MODEL_PATH)
    return _pipeline


def predict_price(make, model, city, year, mileage=None, engine_displacement=None,
                  battery=0.0, is_electric=0):
    if is_electric and engine_displacement is None:
        engine_displacement = 0.0  # electric cars have no engine (same rule as in training)
    row = pd.DataFrame([{
        "make": str(make).strip(),
        "model": str(model).strip(),
        "city": str(city).strip(),
        "car_age": max(REFERENCE_YEAR - int(year), 1),
        "mileage": np.nan if mileage is None else float(mileage),
        "Engine_displacement": np.nan if engine_displacement is None else float(engine_displacement),
        "Battery": float(battery),
        "is_electric": int(is_electric),
        "mileage_missing": int(mileage is None),
        "engine_missing": int(engine_displacement is None),
    }])
    row[["make", "model", "city"]] = row[["make", "model", "city"]].astype(object)
    return float(np.expm1(_get_pipeline().predict(row)[0]))
