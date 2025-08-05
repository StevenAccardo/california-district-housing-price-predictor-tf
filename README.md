# california-district-housing-price-predictor-tf

## Goal
We will use the California Housing Prices from the StatLib repo. This data is based on data from the 1990 CA census. Using this training dataset we will train our model to predict the median house price for other districts.

General Steps

1. A function is called to download and store the dataset locally. The dataset is generally explored to review the data and how it is laid out.
2. The dataset is split into training and test sets using various methods, but with the final split using stratified sampling, which takes a continuous critical feature, like income, and creates categories. This is needed to ensure both the test and training set have equitable distributions of data in each instead of radom distributions that can lead to bias.
3. The dataset is explored using various methods. Histograms, population and price heat maps, feature coorelation examination both in matrix and chart form, as well as deriving new features and examing their relationship to the target also.
4. The dataset is cleaned using various methods.
  1. Imputation is used to fill empty or NAN numerical values with the median value for each attribute of the dataset.
  2. One column holds string data that is categorical, so one hot encoding is used to create additional dummy attributes that will represent the categories numerically.