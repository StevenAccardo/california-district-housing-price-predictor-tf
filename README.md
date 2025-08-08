# california-district-housing-price-predictor-tf

## Goal
We will use the California Housing Prices from the StatLib repo. This data is based on data from the 1990 CA census. Using this training dataset we will train our model to predict the median house price for other districts.

General Steps

1. A function is called to download and store the dataset locally. The dataset is generally explored to review the data and how it is laid out.
2. The dataset is split into training and test sets using various methods, but with the final split using stratified sampling, which takes a continuous critical feature, like income, and creates categories. This is needed to ensure both the test and training set have equitable distributions of data in each instead of radom distributions that can lead to bias.
3. The dataset is explored using various methods. Histograms, population and price heat maps, feature coorelation examination both in matrix and chart form, as well as deriving new features and examing their relationship to the target also.
4. The dataset is preprocessed using sklearn pipe methods and classes.
  1. Imputation is used to fill empty or NAN numerical values with the median value for each attribute of the dataset.
  2. One column holds string data that is categorical, so one hot encoding is used to create additional dummy attributes that will represent the categories numerically.
  3. Logarithmic scaling is applied to several columns to handle heavily scewed data
  4. Several new features are created based off simple math from original columns in the dataset
  5. A custom transformer class is created that leverages k-means algorithm to create clusters and score each record on its proximity to those clusters
5. Once the pre-processing is set up, then the model is trained. Randomized search is used to iterate over various hyperparametes and features, training the model on different combinations. The best model is picked based off of the RMSE. The RMSE for this model and it's parameters is $41,533.
6. This is where I'm stopping this project, but I'll mention possible next steps.


## Possible Next Steps

1. We can explored even more feature and hyperparmeter tuning to try and reduce that RMSE even further. In this project I only used a few variations, so we would likely see some improvement there.
2. Different models can be tested as well, which potentially could yield better results also.
3. This model could be deployed so that in can be used via an API or user interface of some sorts.