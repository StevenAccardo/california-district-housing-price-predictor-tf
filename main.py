import pandas as pd
from data_extract import load_housing_data, explore_data
from split_data import make_stratified_split
from data_discovery import deep_data_explore
from clean_data import preprocessor

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

housing_prepared: pd.DataFrame = preprocessor(housing)

print(housing_prepared.columns)