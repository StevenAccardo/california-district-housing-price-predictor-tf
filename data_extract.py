from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt

# Downloads the dataset if not already in the directory.
def load_housing_data() -> pd.DataFrame:
  tarball_path = Path('datasets/housing.tgz')
  if not tarball_path.is_file():
    Path('datasets').mkdir(parents=True, exist_ok=True)
    url = 'https://github.com/ageron/data/raw/main/housing.tgz'
    urllib.request.urlretrieve(url, tarball_path)
  with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path='datasets')
  return pd.read_csv(Path('datasets/housing/housing.csv'))

def explore_data(data):
  print(data.head())
  print(data.info())
  print(data['ocean_proximity'].value_counts())
  print(data.describe()) # Null values are ignored

  # extra code – the next 5 lines define the default font sizes
  plt.rc('font', size=14)
  plt.rc('axes', labelsize=14, titlesize=14)
  plt.rc('legend', fontsize=14)
  plt.rc('xtick', labelsize=10)
  plt.rc('ytick', labelsize=10)

  # Creates histograms for each one of the numerical columns in the dataset, and sets 50 bins for visually separating the data on the chart.
  data.hist(bins=50, figsize=(12,8))
  plt.show()