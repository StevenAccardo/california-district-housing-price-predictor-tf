import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def clean_data(housing: pd.DataFrame) -> pd.DataFrame:
  # Imputation
  # Will get the median value for all numerical columns/series
  imputer = SimpleImputer(strategy='median')
  # Need to strip out any non-numerical columns
  housing_num = housing.select_dtypes(include=[np.number])
  
  # Calc the median for each attribute
  imputer.fit(housing_num)
  
  # Apply the median to all missing values across the dataset
  X = imputer.transform(housing_num)
  
  # Convert the np array to a dataframe
  housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
  
  from sklearn.preprocessing import OneHotEncoder
  
  cat_encoder = OneHotEncoder(sparse_output=False)
  
  housing_cat = housing['ocean_proximity']
  
  housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

