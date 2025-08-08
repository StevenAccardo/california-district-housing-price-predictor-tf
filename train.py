import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.compose import ColumnTransformer


def train_model(preprocessing: ColumnTransformer) -> RandomizedSearchCV:
  full_pipeline = Pipeline(
    [
      ('preprocessing', preprocessing),
      ('random_forest', RandomForestRegressor(random_state=42))
    ]
  )

  param_distribs = {
    'preprocessing__geo__n_clusters': randint(low=3, high=50),
    'random_forest__max_features': randint(low=2, high=20)
  }

  rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='neg_root_mean_squared_error', random_state=42
  )

  return rnd_search