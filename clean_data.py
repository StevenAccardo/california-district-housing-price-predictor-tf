import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Takes the first column and divides it by the second column for all rows returning their quotient
def column_ratio(X):
  return X[:, [0]] / X[:, [1]]

# Appends the string 'ratio' to each output of the transformer that are passed
# If we don't use thise the FunctionTransformer will just output a placeholder on the end of our new column names
def ratio_name(function_transformer, feature_names_in):
  return ['ratio']  # feature names out

# Creates a custom pipeline that imputes any values in the columns passed to it, invokes a custom transformation, names that transformation, and then scales the output using standardization
# Standardization is good when you have extreme outliers that would normaly squash the data too much.
# Standardization sets the mean value to 0 and the std = 1
def ratio_pipeline():
  return make_pipeline(
    SimpleImputer(strategy='median'),
    FunctionTransformer(column_ratio, feature_names_out=ratio_name),
    StandardScaler()
  )

# This will take any passed columns and return their log as an output.
# The data has positive heavy right tails, so we want to use a more aggresive compression of the data, which is why log is used vs root
# Again, standardization is used here.
def log_pipeline():
  return make_pipeline(
    SimpleImputer(strategy='median'),
    FunctionTransformer(np.log, feature_names_out='one-to-one'),
    StandardScaler()
  )

def cat_pipeline():
  return make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore')
  )

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

# BaseEstimator adds parameter management
# TranformerMixin allows for fit_transform when you have a fit and transform method
# This helps models, especially linear models, better capture spacial patterns by being able to compare districts to centroids.
class ClusterSimilarity(BaseEstimator, TransformerMixin):
  def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
    self.n_clusters = n_clusters
    self.gamma = gamma
    self.random_state = random_state

  # Fit methods learn from the data. Used to compute means, find clusters and etc.
  def fit(self, X, y=None, sample_weight=None):
    # KMeans is an unsupervised clustering algorithm.
    # Picks random center points, assigns each row to its closest point, then adjust the center to the mean of all the points assigned to it. This repeats until the center stops being moved.
    self.kmeans_ = KMeans(self.n_clusters, n_init=10, random_state=self.random_state)
    self.kmeans_.fit(X, sample_weight=sample_weight)
    return self  # always return self!

  # Actually applies the desired transformation
  # Creates 10 columns for each row with a score 0 to 1 showing how similar each row is to each of the clusters from the kmeans algo.
  def transform(self, X):
    return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
  
  def get_feature_names_out(self, names=None):
    return [f'Cluster {i} similarity' for i in range(self.n_clusters)]  
  
def cluster_similiar_pipeline():
  return ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)

# This will catch any left over numerical columns and ensure they are being imputized and scaled to match the rest of the dataset.
def default_num_pipeline():
  return make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
  
def preprocessor(housing: pd.DataFrame) -> pd.DataFrame:
  preprocessing = ColumnTransformer(
    [
      ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
      ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
      ("people_per_house", ratio_pipeline(), ["population", "households"]),
      ("log", log_pipeline(), ["total_bedrooms", "total_rooms", "population",
                              "households", "median_income"]),
      ("geo", cluster_similiar_pipeline(), ["latitude", "longitude"]),
      ("cat", cat_pipeline(), make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline()
  )
  
  preprocessed =  preprocessing.fit_transform(housing)
  
  return pd.DataFrame(
    preprocessed,
    index=housing.index,
    columns=preprocessing.get_feature_names_out()
  )