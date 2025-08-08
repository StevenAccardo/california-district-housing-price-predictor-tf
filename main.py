import pandas as pd
from data_extract import load_housing_data, explore_data
from split_data import make_stratified_split
from data_discovery import deep_data_explore
from data_cleaning import preprocessor
from train import train_model
from sklearn.metrics import root_mean_squared_error

housing = load_housing_data()

# View different aspects of the dataset. Uncomment to use
# explore_data(housing)

train_set, test_set = make_stratified_split(housing, 0.2)
print(len(train_set))
print(len(test_set))

deep_data_explore(train_set)

# Drop our target column, and return a copy of potential features
housing = train_set.drop("median_house_value", axis=1)

# Make a copy of the labels/target to prep for training.
housing_labels = train_set["median_house_value"].copy()
print(housing.columns)

preprocessing = preprocessor()

rnd_search = train_model(preprocessing)
rnd_search.fit(housing, housing_labels)

print(-rnd_search.best_score_)
print(rnd_search.best_params_)

final_model = rnd_search.best_estimator_
feature_importances = final_model['random_forest'].feature_importances_
feature_importances.round(2)

key_features = sorted(
  zip(feature_importances, final_model['preprocessing'].get_feature_names_out()),
  reverse=True
)

print(key_features)

X_test = test_set.drop('median_house_value', axis=1)
y_test = test_set['median_house_value'].copy()

final_predictions = final_model.predict(X_test)

final_rmse = root_mean_squared_error(y_test, final_predictions)
print(final_rmse)