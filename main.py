
from data_extract import load_housing_data, explore_data
from split_data import make_stratified_split
from data_discovery import deep_data_explore


housing = load_housing_data()

# View different aspects of the dataset. Uncomment to use
# explore_data(housing)

train_set, test_set = make_stratified_split(housing, 0.2)
print(len(train_set))
print(len(test_set))

housing_train_set = train_set.copy()
deep_data_explore(housing_train_set)

# Add more features to the dataset to expand potential relationships
# These will be averages
housing_train_set['rooms_per_house'] = housing_train_set['total_rooms'] / housing_train_set['households']
housing_train_set['bedrooms_ratio'] = housing_train_set['total_bedrooms'] / housing_train_set['total_rooms']
housing_train_set['people_per_house'] = housing_train_set['population'] / housing_train_set['households']

# Using same functions, more coefficients now though.
deep_data_explore(housing_train_set)