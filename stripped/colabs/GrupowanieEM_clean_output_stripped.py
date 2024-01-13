from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/PiotrMaciejKowalski/BigData2024Project.git
%cd BigData2024Project
#!git checkout your-branch
%cd ..

!chmod 755 /content/BigData2024Project/src/setup.sh
!/content/BigData2024Project/src/setup.sh

import sys
sys.path.append('/content/BigData2024Project/src')

from start_spark import initialize_spark
initialize_spark()

import pandas as pd
from pyspark.sql import SparkSession

from big_mess.loaders import default_loader, load_single_month, preprocessed_loader


spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()

%%time
nasa_full = default_loader(spark)
nasa_full.count()

nasa_full.show(5)

nasa_full.createOrReplaceTempView("nasa_full")

!pip install datashader


!pip install holoviews hvplot colorcet


!pip install geoviews

from typing import List, Tuple, Optional
import copy
import numpy as np
import matplotlib as mpl
from itertools import combinations
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, jaccard_score, recall_score, roc_auc_score
import datashader as ds
import datashader.transfer_functions as tf
import colorcet as cc
import holoviews as hv
import hvplot.pandas
from holoviews.operation.datashader import datashade
import geoviews as gv
import geoviews.tile_sources as gts
from holoviews import opts
from IPython.display import IFrame
from IPython.core.display import display
from bokeh.plotting import show, output_notebook

nasa2020 = spark.sql(''' SELECT * FROM nasa_full''').where(nasa_full.Year == 2020).drop('Year')
nasa2020.show(5)

nasa_an = preprocessed_loader(spark, "/content/drive/MyDrive/BigMess/NASA/NASA_an.csv")

nasa_an = nasa_an.withColumnRenamed("lon", "lon_sam").withColumnRenamed("lat", "lat_sam")
nasa_an.show(5)

NASA_2020_an = (
     nasa2020
     .join(
        nasa_an,
        [
             nasa2020.lon==nasa_an.lon_sam ,
             nasa2020.lat==nasa_an.lat_sam
         ],
          "inner"
      )
      .drop('lat_sam','lon_sam')
  )

NASA_2020_an.show(5)

NASA_2020_an_pd = NASA_2020_an.toPandas()

NASA_2020_an_pd.head(5)

#agreggating selected features' monthly data over a year:

# a) calculating annual averages:
NASA2020_annual_means = (
                         NASA_2020_an_pd[['lon', 'lat', 'Evap','PotEvap','RootMoist','Rainf','SoilM_100_200cm', 'GVEG']]
                         .groupby( by=['lon', 'lat']
                                  ).mean()
                        ).reset_index()

# b) calculating the total annual rainfall (sum):
NASA2020_rainfall_sum = (
                         NASA_2020_an_pd[['lon', 'lat', 'Rainf']]
                         .groupby( by=['lon', 'lat']
                                  ).sum()
                        .rename(columns={"Rainf": "Annual Rainfall"})
                        .reset_index()
                        )

# c) calculating annual medians for selected features:
NASA2020_medians = (
                   NASA_2020_an_pd[['lon', 'lat', 'PotEvap', 'Evap', 'SoilM_100_200cm', 'AvgSurfT', 'SoilT_40_100cm' ]]
                   .groupby( by=['lon', 'lat']
                            ).median()
                   .rename(columns={'PotEvap': "PotEvap_Median", 'Evap': 'Evap_Median', 'AvgSurfT': 'AvgSurfT_Median',
                                    'SoilT_40_100cm': "SoilT40_100_Median", 'SoilM_100_200cm': 'SoilM_100_200cm_Median'}
                          )
                   .reset_index()
                   )

# Merging into one dataset:
NASA2020_annual_anotated = NASA2020_annual_means.merge(NASA2020_medians, how='inner', on=['lon', 'lat'])


NASA2020_annual_anotated = NASA2020_annual_anotated.merge( NASA2020_rainfall_sum, how='inner', on=['lon', 'lat'])

NASA2020_annual_anotated.head()

def assert_add_column(dataset: pd.DataFrame, feature_name: str, month: int) -> None:
  assert (1<= month <= 12), f'Invalid month number'
  assert feature_name in dataset.columns, f'The attribute "{feature_name}" is not a column in the DataFrame (full_dataset)'


#full_dataset - dataframe with monthly data spanning over a (particular) year
def add_column_to_dataframe(df: pd.DataFrame, full_dataset: pd.DataFrame, feature_name: str, month: int) -> pd.DataFrame:
  # function adds a column to the given dataframe, containing monthly data for the specified feature (feature_name) and month
  assert_add_column(full_dataset, feature_name, month)

  feature_df = full_dataset[['lon', 'lat', feature_name, 'Month']]
  feature_df_month = feature_df[feature_df['Month'] == month]
  feature_df_month = feature_df_month.drop(columns=['Month']).rename(columns = {feature_name: feature_name+str(month)})

  return df.merge(feature_df_month, how='inner', on=['lon', 'lat'])


#adding columns with monthly data for selected features and months
selected_months = [month for month in range(5,10)]
features_names = ['PotEvap', 'Evap', 'Rainf', 'Albedo']

for feature in features_names:
   for month in selected_months:
       NASA2020_annual_anotated = add_column_to_dataframe(NASA2020_annual_anotated, NASA_2020_an_pd, feature, month)

selected_months2 = [month for month in range(3,11)]

for month in selected_months2:
       NASA2020_annual_anotated = add_column_to_dataframe(NASA2020_annual_anotated, NASA_2020_an_pd, 'RootMoist', month)


#adding all columns with "GVEG" monthly data
year = [month for month in range(1,13)]

for month in year:
       NASA2020_annual_anotated = add_column_to_dataframe(NASA2020_annual_anotated, NASA_2020_an_pd, 'GVEG', month)

NASA2020_annual_anotated.head(5)


NASA2020_annual_anotated_ = NASA2020_annual_anotated.drop(columns=['lon', 'lat'])

#data standarization:
scaler = preprocessing.StandardScaler()
standarized = scaler.fit_transform(NASA2020_annual_anotated_)

NASA2020_annual_an_st = pd.DataFrame(standarized, columns=NASA2020_annual_anotated_.columns)
NASA2020_annual_an_st.head()

#dataframe with labels:
labels_an = NASA_2020_an_pd[['lon', 'lat', 'pustynia', 'step']].drop_duplicates()
#desert lables:
desert_label_an = labels_an['pustynia'].tolist()

#Performing Gaussian Mixture algorithm:
gm = GaussianMixture(n_components = 2, n_init = 300, max_iter=200, init_params= 'k-means++', covariance_type='spherical', random_state=42)
gm_result = gm.fit_predict(NASA2020_annual_an_st)

{
'accuracy' : accuracy_score(gm_result, desert_label_an),
'precision' : precision_score(gm_result, desert_label_an),
'jaccard' : jaccard_score(gm_result, desert_label_an),
}

NASA_2020_an_pd.head()

def GM_monthly_data(df: pd.DataFrame, month: int, init_params: str) -> dict:
  #The function performs the Gaussian Mixture algorithm on a dataset limited to data from a given month
  assert (1<=month<=12), f'Invalid month number'

  NASA_monthlyDF = df[df['Month']==month]
  NASA_monthly = NASA_monthlyDF.drop(columns=['Month', 'pustynia'])

  scaler = preprocessing.StandardScaler()
  standarized = scaler.fit_transform(NASA_monthly)
  NASA_monthly_st = pd.DataFrame(standarized, columns=NASA_monthly.columns)

  gm = GaussianMixture(n_components = 2, n_init = 100, max_iter=100, init_params= 'random_from_data', covariance_type='spherical', random_state=42)
  gm_result = gm.fit_predict(NASA_monthly_st)

  labels = NASA_monthlyDF['pustynia'].tolist()
  acc = accuracy_score(gm_result, labels)

  if acc <0.5:
    gm_result = gm_result.tolist()
    gm_result = [(1 - label) for label in gm_result]  #relabeling
    acc = accuracy_score(gm_result, labels)

  pre = precision_score(gm_result, labels)
  jac = jaccard_score(gm_result, labels)

  output = {'GM_result': gm_result, 'acc': acc, 'pre': pre, 'jac': jac}
  return output

NASA_2020_an_pd = NASA_2020_an_pd.drop(columns=['step', 'SoilT_40_100cm', 'SoilM_100_200cm'])

# GaussianMixture with init_param = "random_from_data"
for i in range(12):
  gm = GM_monthly_data(NASA_2020_an_pd, i+1, "random_from_data")
  print("Accuracy dla danych z miesiąca ", i+1, " wynosi ", round(gm['acc'], 3),
        " , Precision: ", round(gm['pre'], 3), ", Jaccard score: ", round(gm['jac'], 3) )


# GaussianMixture with init_param = "k-means++"
for i in range(12):
  gm = GM_monthly_data(NASA_2020_an_pd, i+1, "k-means++")
  print("Accuracy dla danych z miesiąca ", i+1, " wynosi ", round(gm['acc'], 3),
        " , Precision: ", round(gm['pre'], 3), ", Jaccard score: ", round(gm['jac'], 3) )


labels1 = np.zeros(500)
months = [5,6,7,8,9,11]
for i in range(6):
  gm = GM_monthly_data(NASA_2020_an_pd, months[i], 'k-means++')
  acc, pre, jac, labels2 = gm['acc'], gm['pre'], gm['jac'], gm['GM_result']
  for j in range(len(labels1)):
     labels1[j] = labels1[j] + labels2[j]

for j in range(len(labels1)):
  if labels1[j]>=4:
     labels1[j] = 1
  else:
     labels1[j] = 0

metrics = {
'accuracy' : accuracy_score(labels1, desert_label_an),
'precision' : precision_score(labels1, desert_label_an),
'jaccard' : jaccard_score(labels1, desert_label_an),
}
print(metrics)

NASA2020_full = nasa2020.toPandas()

#agreggating selected features' monthly data over a year:

# a) calculating annual averages:
NASA2020FULL_annual_means = (
                         NASA2020_full[['lon', 'lat', 'Evap','PotEvap','RootMoist','Rainf','SoilM_100_200cm', 'GVEG']]
                         .groupby( by=['lon', 'lat']
                                  ).mean()
                        ).reset_index()

# b) calculating the total annual rainfall (sum):
NASA2020FULL_rainfall_sum = (
                         NASA2020_full[['lon', 'lat', 'Rainf']]
                         .groupby( by=['lon', 'lat']
                                  ).sum()
                        .rename(columns={"Rainf": "Annual Rainfall"})
                        .reset_index()
                        )

# c) calculating annual medians for selected features:
NASA2020FULL_annual_medians = (
                   NASA2020_full[['lon', 'lat', 'PotEvap', 'Evap', 'SoilM_100_200cm', 'AvgSurfT', 'SoilT_40_100cm' ]]
                   .groupby( by=['lon', 'lat']
                            ).median()
                   .rename(columns={'PotEvap': "PotEvap_Median", 'Evap': 'Evap_Median', 'AvgSurfT': 'AvgSurfT_Median',
                                    'SoilT_40_100cm': "SoilT40_100_Median",'SoilM_100_200cm': 'SoilM_100_200cm_Median'}
                          )
                   .reset_index()
                   )

# Merging into one dataset:
NASA2020_annual_full = NASA2020FULL_annual_means.merge(NASA2020FULL_annual_medians, how='inner', on=['lon', 'lat'])

NASA2020_annual_full = NASA2020_annual_full.merge(NASA2020FULL_rainfall_sum, how='inner', on=['lon', 'lat'])

NASA2020_annual_full_2 = copy.deepcopy(NASA2020_annual_full)

NASA2020_annual_full.head(5)

NASA2020_annual_full_ = NASA2020_annual_full.drop(columns=['lon', 'lat'])

#data standarization:
scaler = preprocessing.StandardScaler()
standarized = scaler.fit_transform(NASA2020_annual_full_)
st_NASA2020_annual_full = pd.DataFrame(standarized, columns=NASA2020_annual_full_.columns)
st_NASA2020_annual_full.head()

GM = GaussianMixture(n_components = 2, n_init = 200, max_iter=200, init_params= 'random_from_data', covariance_type='spherical', random_state=42)
gm_result = GM.fit_predict(st_NASA2020_annual_full)

NASA2020_annual_full['label'] = list(gm_result)
anotated_subset = NASA2020_annual_full[['lon', 'lat', 'label']].merge(labels_an[['lon', 'lat', 'pustynia']], on=['lon', 'lat'], how='inner')

metrics = {
'accuracy': accuracy_score(list(anotated_subset['label']), list(anotated_subset['pustynia'])),
'precision': precision_score(list(anotated_subset['label']), list(anotated_subset['pustynia'])),
'recall': round(recall_score(list(anotated_subset['label']), list(anotated_subset['pustynia'])), 4),
'jaccard_score' : round(jaccard_score(list(anotated_subset['label']), list(anotated_subset['pustynia'])), 4)
}
print(metrics)

#adding columns with monthly data for selected features and months
selected_months = [month for month in range(5,10)]
features_names = ['PotEvap', 'Evap', 'Rainf', 'Albedo']

for feature in features_names:
   for month in selected_months:
       NASA2020_annual_full_2 = add_column_to_dataframe(NASA2020_annual_full_2, NASA2020_full, feature, month)

selected_months2 = [month for month in range(3,11)]

for month in selected_months2:
       NASA2020_annual_full_2 = add_column_to_dataframe(NASA2020_annual_full_2, NASA2020_full, 'RootMoist', month)


#adding all columns with "GVEG" monthly data
year = [month for month in range(1,13)]

for month in year:
       NASA2020_annual_full_2 = add_column_to_dataframe(NASA2020_annual_full_2, NASA2020_full, 'GVEG', month)


NASA2020_annual_full_2.head()

NASA2020_annual_full_2_ = NASA2020_annual_full_2.drop(columns=['lon', 'lat'])

#data standarization:
scaler = preprocessing.StandardScaler()
standarized = scaler.fit_transform(NASA2020_annual_full_2_)
st_NASA2020_annual_full2 = pd.DataFrame(standarized, columns=NASA2020_annual_full_2_.columns)
st_NASA2020_annual_full2.head()

GM = GaussianMixture(n_components = 2, n_init = 200, max_iter=200, init_params= 'random_from_data', covariance_type='spherical', random_state=42)
gm_result = GM.fit_predict(st_NASA2020_annual_full2)

NASA2020_annual_full_2['label'] = gm_list
anotated_subset = NASA2020_annual_full_2[['lon', 'lat', 'label']].merge(labels_an[['lon', 'lat', 'pustynia']], on=['lon', 'lat'], how='inner')

metrics = {
'accuracy': accuracy_score(list(anotated_subset['label']), list(anotated_subset['pustynia'])),
'precision': precision_score(list(anotated_subset['label']), list(anotated_subset['pustynia'])),
'recall': round(recall_score(list(anotated_subset['label']), list(anotated_subset['pustynia'])), 4),
'jaccard-score': round(jaccard_score(list(anotated_subset['label']), list(anotated_subset['pustynia'])), 4),
'ROC-AUC': round(roc_auc_score(list(anotated_subset['label']), list(anotated_subset['pustynia'])), 4)
}

print(metrics)

gm_list = gm_result.tolist()

deserts = {
    'number of deserts predicted': sum(gm_list),
    'percentage deserts predicted': sum(gm_list)/len(NASA2020_annual_full)
}
print(deserts)

def GM_monthly_data(month: int, df: pd.DataFrame, evaluation_subset_labels: pd.DataFrame) -> dict:
  assert (1<=month<=12), f'Invalid month number'

  MonthDF = df[df['Month']==month]     #selecting data from given month
  MonthDF_ = MonthDF.drop(columns=['lon', 'lat', 'Month'])

  scaler = preprocessing.StandardScaler()   #data standarization
  standarized = scaler.fit_transform(MonthDF_)
  st_MonthDF_ = pd.DataFrame(standarized, columns=MonthDF_.columns)

  gm = GaussianMixture(n_components = 2, n_init = 200, max_iter=200, init_params= 'random_from_data',
                       covariance_type='spherical', random_state=42)
  gm_result = gm.fit_predict(st_MonthDF_)
  gm_list = gm_result.tolist()

  if (sum(gm_list)/len(MonthDF)) >0.5:
     gm_list = [(1-label) for label in gm_list] #relabeling

  deserts_percentage = round(sum(gm_list)/len(MonthDF), 4)

  GM_labels = pd.DataFrame({'lon': MonthDF['lon'].tolist(), 'lat': MonthDF['lat'].tolist(), 'label': gm_list })
  anotated_subset = GM_labels.merge( evaluation_subset_labels[['lon', 'lat', 'pustynia']],
                                              on=['lon', 'lat'], how='inner')
  accuracy_on_anotated = accuracy_score(anotated_subset['pustynia'], anotated_subset['label'] )
  precision_on_anotated = precision_score(anotated_subset['pustynia'], anotated_subset['label'] )
  roc_auc_on_anotated = roc_auc_score(anotated_subset['pustynia'], anotated_subset['label'])

  output = {'gm_labels': gm_list, 'deserts percentage': deserts_percentage,'accuracy on anotated': accuracy_on_anotated,
            'precision on anotated' : precision_on_anotated, "ROC-auc on anotated": roc_auc_on_anotated }

  return output

#removing two particular columns ('SoilT_40_100cm' and 'SoilM_100_200cm') from the dataset:
#removing those columns from the dataset resulted in significantly better algorithm performance
NASA2020_full_reduced = NASA2020_full.drop(columns = ['SoilT_40_100cm', 'SoilM_100_200cm'])

clustering_labels = []
month, deserts_per, acc, pre, roc_auc = [], [], [], [], []


for i in range(12):
  gm = GM_monthly_data(i+1, NASA2020_full_reduced, labels_an)
  clustering_labels.append(gm['gm_labels'])
  month.append(i+1)
  deserts_per.append(gm['deserts percentage'])
  acc.append(gm['accuracy on anotated'])
  pre.append(gm['precision on anotated'])
  roc_auc.append(gm['ROC-auc on anotated'])

evaluation = pd.DataFrame(data = {"Month data": month, "predicted deserts' percentage": deserts_per,
            "accuracy on anotated subset": acc, "precision on anotated subset": pre, "ROC-AUC on anotated subset": roc_auc  })


evaluation

def get_colormap(values: list, colors_palette: list, name = 'custom'):
    values = np.sort(np.array(values))
    values = np.interp(values, (values.min(), values.max()), (0, 1))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(name, list(zip(values, colors_palette)))
    return cmap

def plot_map(df: pd.DataFrame, parameter_name: str, colormap: mpl.colors.LinearSegmentedColormap,
             point_size: int = 8, width: int = 900, height: int = 600, alpha: float = 1,
             bgcolor: str = 'white'):

    gdf = gv.Points(df, ['lon', 'lat'], [parameter_name]) # obiekt zawierający punkty
    tiles = gts.OSM # wybór mapy tła, w tym wypadku OpenStreetMap

    # łączenie mapy tła z punktami i ustawienie wybranych parametrów wizualizacji
    map_with_points = tiles * gdf.opts(
        color=parameter_name,
        cmap=colormap,
        size=point_size,
        width=width,
        height=height,
        colorbar=False,
        toolbar='above',
        tools=['hover', 'wheel_zoom', 'reset'],
        alpha=alpha
    )

    return hv.render(map_with_points)

colormap = get_colormap([0, 1], ['green', 'yellow'])
output_notebook()
show(plot_map(df=NASA2020_annual_full[['lon', 'lat', 'label']], parameter_name='label', colormap=colormap, alpha=0.5))

for i in range(12):
  print("Wizualizacja grupowania na danych z miesiąca: ", i+1)
  nasa_month = NASA2020_full[NASA2020_full['Month']==i+1]
  df = pd.DataFrame(nasa_month[['lon','lat']])
  df['label'] = clustering_labels[i]
  colormap = get_colormap([0, 1], ['green', 'yellow'])
  output_notebook()
  show(plot_map(df=df, parameter_name='label', colormap=colormap, alpha=0.5))

column_names = NASA2020_full.drop(columns=['lon', 'lat', 'Month']).columns

for column in column_names:
  NASA_single = pd.DataFrame(NASA2020_full[[column, 'Month', 'lon', 'lat']])
  gmm = GM_monthly_data(5, NASA_single, labels_an)
  labels = gmm.pop('gm_labels')
  print("Results for clustering based on the ", column, "column:")
  print(gmm)

  nasa_month = NASA2020_full[NASA2020_full['Month']==5]
  df = pd.DataFrame(nasa_month[['lon','lat']])
  df['label'] = labels
  colormap = get_colormap([0, 1], ['green', 'yellow'])
  output_notebook()
  show(plot_map(df=df, parameter_name='label', colormap=colormap, alpha=0.5))


for pair in combinations(column_names, 2):
  columns = list(pair)
  NASA_2features = pd.DataFrame(NASA2020_full[[columns[0], columns[1], 'Month', 'lon', 'lat']])
  gmm = GM_monthly_data(5, NASA_2features, labels_an)
  labels = gmm.pop('gm_labels')
  print("Results for clustering based on the following columns : ", columns)
  print(gmm)

  nasa_month = NASA2020_full[NASA2020_full['Month']==5]
  df = pd.DataFrame(nasa_month[['lon','lat']])
  df['label'] = labels
  colormap = get_colormap([0, 1], ['green', 'yellow'])
  output_notebook()
  show(plot_map(df=df, parameter_name='label', colormap=colormap, alpha=0.5))

