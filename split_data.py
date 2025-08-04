import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# # Custom function for splitting the data randomly
# # Takes the dataset, a test ratio. Returns the dataset split into a training set and a test set.
# def shuffle_and_split_data(data, test_ratio) -> tuple[DataFrame, DataFrame]:
#   # Ensures any random generation is repeatable as long as the dataset remains static (A case I'm leaning on for this short term project.)
#   np.random.seed(42)
#   # Takes the # of rows in the dataframe and uses that number to create a shuffled list of indices 0 to n - 1.
#   # Import to shuffle the data especially when the data is situated in categories such as districts (lat, long) in this case. This removes bias from the training set.
#   shuffled_indices = np.random.permutation(len(data))
#   test_set_size = int(len(data) * test_ratio)
#   # Returns first portion of the DF
#   test_indices = shuffled_indices[:test_set_size]
#   # Returns the rest of the DF
#   train_indices = shuffled_indices[test_set_size:]
#   return data.iloc[train_indices], data.iloc[test_indices]

# # sklearn function for doing a similar random split above
# from sklearn.model_selection import train_test_split
# train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# Breaks a key feature for the model into stratified categories so that our test and training data can more accurately represent the data
def make_stratified_split(data: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
  data['income_cat'] = pd.cut(data['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
  strat_train_set, strat_test_set = train_test_split(data, test_size=test_ratio, stratify=data['income_cat'], random_state=42)
  
  # Drops the stratified categories as we won't need them any longer when training or testing this model.
  for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    
  return strat_train_set, strat_test_set