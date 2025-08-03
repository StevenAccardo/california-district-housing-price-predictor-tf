
from data_extract import load_housing_data, explore_data
from split_data import shuffle_and_split_data


housing = load_housing_data()

# View different aspects of the dataset. Uncomment to use
# explore_data(housing)

train_set, test_set = shuffle_and_split_data(housing, 0.2)
print(len(train_set))
print(len(test_set))