!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q dlcdn.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
!tar xf spark-3.5.0-bin-hadoop3.tgz
!pip install -q findspark

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.5.0-bin-hadoop3"

import findspark
findspark.init()

!pip install datashader


!pip install holoviews hvplot colorcet


!pip install geoviews

from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib as mpl
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, jaccard_score, recall_score
import datashader as ds
import datashader.transfer_functions as tf
import colorcet as cc
import holoviews as hv
from holoviews.operation.datashader import datashade
import geoviews as gv
import geoviews.tile_sources as gts
from holoviews import opts
from IPython.display import IFrame
from IPython.core.display import display
from bokeh.plotting import show, output_notebook

from pyspark.sql import SparkSession, DataFrame as SparkDataFrame, functions as F
from pyspark.sql.types import IntegerType, FloatType, StringType, StructType, StructField, Row

from google.colab import drive

spark = (
         SparkSession.builder
        .master("local")
        .appName("Colab")
        .config('spark.ui.port', '4050')
        .getOrCreate()
)
conf = spark.sparkContext._conf.setAll([('spark.executor.memory', '100g'), ('spark.driver.memory','64g')])
spark.conf.set("park.driver.maxResultSize", "80g")

spark.conf.set('spark.sql.execution.arrow.enabled', 'true')

drive.mount('/content/drive')

columns = ['lon', 'lat', 'Date', 'Rainf', 'Evap', 'AvgSurfT', 'Albedo','SoilT_40_100cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilM_100_200cm']

# Utworzenie schematu określającego typ zmiennych
schema = StructType()
for i in columns:
  if i == "Date":
    schema = schema.add(i, IntegerType(), True)
  else:
    schema = schema.add(i, FloatType(), True)

%%time
# Wczytanie zbioru Nasa w sparku

nasa = spark.read.format('csv').option("header", True).schema(schema).load('/content/drive/MyDrive/BigMess/NASA/NASA.csv')

nasa = (
    nasa
    .withColumn('Year', (F.col('Date') / 100).cast('int'))
    .withColumn('Month', F.col('Date') % 100)
    .drop('Date')
)
nasa.show(5)

nasa.createOrReplaceTempView("nasa")

nasa2020 = spark.sql(''' SELECT * FROM nasa''').where(nasa.Year == 2020).drop('Year')
nasa2020.show(5)

NASA_sample_annotated = pd.read_csv('/content/drive/MyDrive/BigMess/NASA/NASA_an.csv',sep=';')
NASA_sample_annotated = spark.createDataFrame(NASA_sample_annotated).withColumnRenamed("lat", "lat_sam").withColumnRenamed("lon", "lon_sam")
NASA_sample_annotated.show(5)

 NASA_2020_an = nasa2020.join( NASA_sample_annotated,
                [nasa2020.lon==NASA_sample_annotated.lon_sam , nasa2020.lat==NASA_sample_annotated.lat_sam],
                "inner").drop('lat_sam').drop('lon_sam')

 NASA_2020_an.show(5)

NASA_2020_set = NASA_2020_an.toPandas()

NASA_2020_set.head(5)

NASA2020_means = (NASA_2020_set[['lon', 'lat', 'Evap','PotEvap','RootMoist','Rainf','SoilM_100_200cm', 'GVEG']].groupby(
    by=['lon', 'lat']).mean()).reset_index()

#Roczna suma opadów:
NASA2020_sum = (NASA_2020_set[['lon', 'lat', 'Rainf']].groupby(by=['lon', 'lat']).sum()).rename(columns={"Rainf": "Annual Rainfall"}).reset_index()

#Kolumny ze srednimi miesiecznymi indeksu zielonej roslinnosci:
GVEG_columns = pd.DataFrame({'lon':[], 'lat':[], 'GVEG1':[], 'GVEG2':[], 'GVEG3':[], 'GVEG4':[], 'GVEG5':[], 'GVEG6':[], 'GVEG7':[],
                             'GVEG8':[], 'GVEG9':[], 'GVEG10': [], 'GVEG11':[], 'GVEG12':[] })

coordinates = NASA_2020_set[['lon', 'lat']]
coordinates = coordinates.drop_duplicates()
NASA2020_GVEG = NASA_2020_set[['lon', 'lat', 'GVEG', 'Month']]

for i in coordinates.index:
  data = NASA2020_GVEG[(NASA2020_GVEG['lon']==coordinates.at[i,'lon'])&(NASA2020_GVEG['lat']==coordinates.at[i, 'lat'])]
  data = data.sort_values(by=['Month'])
  GVEG = data['GVEG'].tolist()

  row = {'lon': coordinates.at[i, 'lon'], 'lat': coordinates.at[i, 'lat'], 'GVEG1': GVEG[0], 'GVEG2': GVEG[1], 'GVEG3': GVEG[2],
         'GVEG4': GVEG[3], 'GVEG5': GVEG[4], 'GVEG6': GVEG[5], 'GVEG7': GVEG[6], 'GVEG8': GVEG[7], 'GVEG9': GVEG[8], 'GVEG10': GVEG[9],
         'GVEG11': GVEG[10], 'GVEG12': GVEG[11]}

  GVEG_columns.loc[len(GVEG_columns)] = row

NASA2020an = NASA2020_means.merge(NASA2020_sum, how='inner', on=['lon', 'lat']).merge(GVEG_columns, how='inner', on=['lon', 'lat'])

#Mediany dla wybranych cech:
NASA2020_median = (NASA_2020_set[['lon', 'lat', 'PotEvap', 'Evap', 'SoilM_100_200cm', 'AvgSurfT', 'SoilT_40_100cm' ]]).rename(
    columns={'PotEvap': "PotEvap_Median", 'Evap': 'Evap_Median', 'AvgSurfT': 'AvgSurfTmedian', 'SoilT_40_100cm': "SoilT40_100_Median",
            'SoilM_100_200cm': 'SoilM_100_200cm_Median'}).groupby(by=['lon', 'lat']).median().reset_index()

NASA2020an = NASA2020an.merge(NASA2020_median, how='inner', on=['lon', 'lat'])

#tabela z etykietami:
labels = NASA_2020_set[['lon', 'lat', 'pustynia', 'step']].drop_duplicates()

def add_column(feature: str, month: int, dataset: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
  coordinates = dataset[['lon', 'lat']]
  coordinates = coordinates.drop_duplicates()
  feature_df = dataset[['lon', 'lat', feature, 'Month']]
  feature_df_m = feature_df[feature_df['Month'] == month]
  feature_df_m = feature_df_m.drop(columns=['Month']).rename(columns = {feature: feature+str(month)})

  return data.merge(feature_df_m, how='inner', on=['lon', 'lat'])


NASA2020an = add_column('PotEvap', 5, NASA_2020_set, NASA2020an)
NASA2020an = add_column('PotEvap', 6, NASA_2020_set, NASA2020an)
NASA2020an = add_column('PotEvap', 7, NASA_2020_set, NASA2020an)
NASA2020an = add_column('PotEvap', 8, NASA_2020_set, NASA2020an)
NASA2020an = add_column('PotEvap', 9, NASA_2020_set, NASA2020an)

NASA2020an = add_column('Evap', 5, NASA_2020_set, NASA2020an)
NASA2020an = add_column('Evap', 6, NASA_2020_set, NASA2020an)
NASA2020an = add_column('Evap', 7, NASA_2020_set, NASA2020an)
NASA2020an = add_column('Evap', 8, NASA_2020_set, NASA2020an)
NASA2020an = add_column('Evap', 9, NASA_2020_set, NASA2020an)

NASA2020an = add_column('Rainf', 5, NASA_2020_set, NASA2020an)
NASA2020an = add_column('Rainf', 6, NASA_2020_set, NASA2020an)
NASA2020an = add_column('Rainf', 7, NASA_2020_set, NASA2020an)
NASA2020an = add_column('Rainf', 8, NASA_2020_set, NASA2020an)
NASA2020an = add_column('Rainf', 9, NASA_2020_set, NASA2020an)

NASA2020an = add_column('Albedo', 5, NASA_2020_set, NASA2020an)
NASA2020an = add_column('Albedo', 6, NASA_2020_set, NASA2020an)
NASA2020an = add_column('Albedo', 7, NASA_2020_set, NASA2020an)
NASA2020an = add_column('Albedo', 8, NASA_2020_set, NASA2020an)
NASA2020an = add_column('Albedo', 9, NASA_2020_set, NASA2020an)

for i in range(4, 10):
    NASA2020an = add_column('RootMoist', i , NASA_2020_set, NASA2020an)

NASA2020an.head(5)


NASA2020features = NASA2020an.drop(columns=['lon', 'lat'])

#standaryzujemy dane:
scaler = preprocessing.StandardScaler()
scaler.fit(NASA2020features)
standarized = scaler.transform(NASA2020features)

st_NASA2020 = pd.DataFrame(standarized, columns=NASA2020features.columns)
st_NASA2020.head()

gm = GaussianMixture(n_components = 2, n_init = 300, max_iter=200, init_params= 'k-means++', covariance_type='spherical', random_state=42)
gm_result = gm.fit_predict(st_NASA2020)

labels_ = labels['pustynia'].tolist()

accuracy_score(gm_result, labels_)

precision_score(gm_result, labels_)

jaccard_score(gm_result, labels_)

stNASA2020_ver2 = st_NASA2020.drop(columns = ['GVEG1', 'GVEG2', 'GVEG12'])

gm = GaussianMixture(n_components = 2, n_init = 300, max_iter=200, init_params= 'random_from_data', covariance_type='spherical', random_state=42)
gm_result = gm.fit_predict(stNASA2020_ver2)

{ 
'accuracy' : accuracy_score(gm_result, labels_),
'precision' : precision_score(gm_result, labels_),
'jaccard' : jaccard_score(gm_result, labels_),
}


NASA_2020_set

def gmm_monthly_data(month: int, df: pd.DataFrame) -> Tuple[float, float, float, List[int]]:

  NASA_M_2020 = df[df['Month']==month]
  NASA_month = NASA_M_2020.drop(columns=['lon', 'lat', 'pustynia', 'step', 'Month', 'SoilT_40_100cm', 'SoilM_100_200cm'])
  scaler = preprocessing.StandardScaler()
  scaler.fit(NASA_month)
  standarized = scaler.transform(NASA_month)
  st_NASA_month = pd.DataFrame(standarized, columns=NASA_month.columns)
  gm = GaussianMixture(n_components = 2, n_init = 100, max_iter=100, init_params= 'random_from_data', covariance_type='spherical')
  gm_result = gm.fit_predict(st_NASA_month)
  label = NASA_M_2020['pustynia'].tolist()
  acc = accuracy_score(gm_result, label)

  if acc <0.5:
    acc = 1 - acc
    gm_result = gm_result.tolist()
    for i in range(len(gm_result)):
      gm_result[i] = 1 - gm_result[i]

  pre = precision_score(gm_result, label)
  jac = jaccard_score(gm_result, label)
  return (round(acc, 3), round(pre, 3), round(jac, 3), gm_result)

for i in range(12):
  acc, pre, jac, labels = gmm_monthly_data(i+1, NASA_2020_set)
  print("Accuracy dla danych z miesiąca ", i+1, " wynosi ", acc, " , Precision: ", pre, ", Jaccard score: ", jac )


labels1 = np.zeros(500)
months = [5,6,7,8,9,11]
for i in range(6):
  acc, pre, jac, labels2 = gmm_monthly_data(months[i], NASA_2020_set)
  for j in range(len(labels1)):
     labels1[j] = labels1[j] + labels2[j]

for j in range(len(labels1)):
  if labels1[j]>=4:
     labels1[j] = 1
  else:
     labels1[j] = 0

accuracy_score(labels1, labels_)

precision_score(labels1, labels_)

jaccard_score(labels1, labels_)

NASA2020_full = nasa2020.toPandas()

#Srednie dla wybranych cech:
NASA2020_means = (NASA2020_full[['lon', 'lat', 'Evap','PotEvap','RootMoist','Rainf','SoilM_100_200cm', 'GVEG']].groupby(
    by=['lon', 'lat']).mean()).reset_index()

#Roczna suma opadów:
NASA2020_sum = (NASA2020_full[['lon', 'lat', 'Rainf']].groupby(by=['lon', 'lat']).sum()).rename(columns={"Rainf": "Annual Rainfall"}).reset_index()

NASA2020 = NASA2020_means.merge(NASA2020_sum, how='inner', on=['lon', 'lat'])

#Mediany dla wybranych cech:
NASA2020_median = (NASA2020_full[['lon', 'lat', 'PotEvap', 'Evap', 'SoilM_100_200cm', 'AvgSurfT', 'SoilT_40_100cm' ]]).rename(
    columns={'PotEvap': "PotEvap_Median", 'Evap': 'Evap_Median', 'AvgSurfT': 'AvgSurfTmedian', 'SoilT_40_100cm': "SoilT40_100_Median",
            'SoilM_100_200cm': 'SoilM_100_200cm_Median'}).groupby(by=['lon', 'lat']).median().reset_index()

NASA2020 = NASA2020.merge(NASA2020_median, how='inner', on=['lon', 'lat'])


default_months = list(range(5, 10))
features_interesting_months = {
'RootMoist' : list(range(4, 10)),
'GVEG' : list(range(4, 11)),
'Rainf' : default_month, 
'PotEvap' : default_month, 
'Evap' : default_month, 
'Albedo' : default_month,

 }
 
 for feature, months in features_interesting_months.items():
    for month in months:
       NASA2020 = add_column(feature, month , NASA2020_full, NASA2020)
 

NASA2020.head()

NASA2020feat = NASA2020.drop(columns=['lon', 'lat'])

#standaryzujemy dane:
scaler = preprocessing.StandardScaler()
scaler.fit(NASA2020feat)
standarized = scaler.transform(NASA2020feat)

st_NASA2020 = pd.DataFrame(standarized, columns=NASA2020feat.columns)
st_NASA2020.head()

gm = GaussianMixture(n_components = 2, n_init = 200, max_iter=200, init_params= 'random_from_data', covariance_type='spherical', random_state=22)
gm_result = gm.fit_predict(st_NASA2020)

gm_list = gm_result.tolist()
number_of_label1 = sum(gm_list)
print(number_of_label1)

percentage_deserts_predicted = sum(gm_list)/len(NASA2020)
print(percentage_deserts_predicted)

NASA2020['label'] = gm_list
anotated_unique = NASA_2020_set[NASA_2020_set['Month']==1]
anotated_subset = NASA2020[['lon', 'lat', 'label']].merge(anotated_unique[['lon', 'lat', 'pustynia']], on=['lon', 'lat'], how='inner')

accuracy_score(list(anotated_subset['label']), list(anotated_subset['pustynia']))

precision_score(list(anotated_subset['label']), list(anotated_subset['pustynia']))

recall_score(list(anotated_subset['label']), list(anotated_subset['pustynia']))

def GM_monthly_data(month: int, df: pd.DataFrame) -> Tuple[List[int], float, float, float]:

  NASA_M_2020 = df[df['Month']==month]
  NASA_month = NASA_M_2020.drop(columns=['lon', 'lat', 'Month', 'SoilT_40_100cm', 'SoilM_100_200cm'])
  scaler = preprocessing.StandardScaler()
  scaler.fit(NASA_month)
  standarized = scaler.transform(NASA_month)
  st_NASA_month = pd.DataFrame(standarized, columns=NASA_month.columns)
  gm = GaussianMixture(n_components = 2, n_init = 200, max_iter=200, init_params= 'random_from_data', covariance_type='spherical')
  gm_result = gm.fit_predict(st_NASA_month)

  gm_list = gm_result.tolist()
  if (sum(gm_list)/len(NASA2020)) >0.5:
    for i in range(len(gm_list)):
        gm_list[i] = 1 - gm_list[i]

  deserts_predicted_rate = round(sum(gm_list)/len(NASA2020), 4)
  coordinates_labels = pd.DataFrame({'lon': NASA_M_2020['lon'].tolist(), 'lat': NASA_M_2020['lat'].tolist(), 'label': gm_list })
  anotated_subset = coordinates_labels.merge(
      anotated_unique[['lon', 'lat', 'pustynia']], on=['lon', 'lat'], how='inner')
  accuracy_on_anotated = accuracy_score(anotated_subset['pustynia'], anotated_subset['label'] )
  precision_on_anotated = precision_score(anotated_subset['pustynia'], anotated_subset['label'] )


  return gm_list, deserts_predicted_rate, round(accuracy_on_anotated, 3), round(precision_on_anotated, 4)

clustering_labels = []

for i in range(12):
  gm, rate, acc, prec = GM_monthly_data(i+1, NASA2020_full)
  clustering_labels.append(gm)
  print("Clustering based on data from month ", i+1, ": predicted deserts' percentage: ", rate, ", accuracy on anotated subset: ", acc,
        ", precision on anotated subset: ", prec )

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

display(IFrame("https://www.google.com/maps/embed?pb=!1m14!1m12!1m3!1d13982681.959428234!2d-98.66341902257437!3d38.39997874427714!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!5e1!3m2!1spl!2spl!4v1703000232420!5m2!1spl!2spl", '800px', '500px'))

colormap = get_colormap([0, 1], ['green', 'yellow'])
output_notebook()
show(plot_map(df=NASA2020[['lon', 'lat', 'label']], parameter_name='label', colormap=colormap, alpha=0.5))

for i in range(12):
  print("Wizualizacja grupowania na danych z miesiąca: ", i+1)
  nasa_month = NASA2020_full[NASA2020_full['Month']==i+1]
  df = pd.DataFrame(nasa_month[['lon','lat']])
  df['label'] = clustering_labels[i]
  colormap = get_colormap([0, 1], ['green', 'yellow'])
  output_notebook()
  show(plot_map(df=df, parameter_name='label', colormap=colormap, alpha=0.5))
