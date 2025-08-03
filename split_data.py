import numpy as np
from pandas import DataFrame

# Takes the dataset, a test ratio. Returns the dataset split into a training set and a test set.
def shuffle_and_split_data(data, test_ratio) -> tuple[DataFrame, DataFrame]:
  # Takes the # of rows in the dataframe and uses that number to create a shuffled list of indices 0 to n - 1.
  # Import to shuffle the data especially when the data is situated in categories such as districts (lat, long) in this case. This removes bias from the training set.
  shuffled_indices = np.random.permutation(len(data))
  test_set_size = int(len(data) * test_ratio)
  # Returns first portion of the DF
  test_indices = shuffled_indices[:test_set_size]
  # Returns the rest of the DF
  train_indices = shuffled_indices[test_set_size:]
  return data.iloc[train_indices], data.iloc[test_indices]