import pandas as pd
import matplotlib.pyplot as plt

def deep_data_explore(housing: pd.DataFrame):
  # Population size directly related to circle diameter, and price directly related to color.
  housing.plot(
    kind='scatter',
    x='longitude',
    y='latitude',
    grid=True,
    s=housing['population'] / 100,
    label = 'Population',
    c='median_house_value',
    cmap='jet',
    colorbar=True,
    legend=True,
    # The argument sharex=False fixes a display bug: without it, the x-axis values and label are not displayed.
    sharex=False, figsize=(10, 7)
  )
  plt.show()
  
  # Creates a correlation matrix that holds the pair-wise correlation coefficients for each numerical column in the DF
  # Use the Pearson's r, which ranges from -1 to 1 with -1 being inversly proportional and 1 being directly proportional
  corr_matrix = housing.corr(numeric_only=True)
  
  # Prints the correlation coefficients for each column relative to median_house_value
  med_house_value_coefficients = corr_matrix['median_house_value'].sort_values(ascending=False)
  print(med_house_value_coefficients)
  
  # Also a way of showing which features more strongly coorelate to the median_house_value
  from pandas.plotting import scatter_matrix

  attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
  scatter_matrix(housing[attributes], figsize=(12, 8))
  plt.show()
  