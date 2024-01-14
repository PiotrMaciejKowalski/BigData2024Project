"""
# Worst Case Scenario
"""

!pip install pyspark
!pip install datashader
!pip install holoviews hvplot colorcet
!pip install geoviews

import pyspark
import pickle
import numpy as np
import pandas as pd
import geoviews as gv
import colorcet as cc
import holoviews as hv
import datashader as ds
import matplotlib as mpl
import geoviews.tile_sources as gts
import datashader.transfer_functions as tf

from google.colab import drive
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType, StringType, StructType
from pyspark.sql.functions import col
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from holoviews.operation.datashader import datashade
from holoviews import opts
from IPython.display import IFrame
from IPython.core.display import display
from bokeh.plotting import show, output_notebook
from sklearn.mixture import GaussianMixture


# tworzenie sesji w Sparku
spark = SparkSession.builder.appName('SparkWindows').getOrCreate()


# wczytanie danych z google drive
drive.mount('/content/drive')

columns = ['lon', 'lat', 'Date', 'Rainf', 'Evap', 'AvgSurfT', 'Albedo','SoilT_40_100cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilM_100_200cm']

# Utworzenie schematu okreslajacego typ zmiennych
schema = StructType()
for i in columns:
  if i == "Date":
    schema = schema.add(i, IntegerType(), True)
  else:
    schema = schema.add(i, FloatType(), True)

nasa = spark.read.format('csv').option("header", True).schema(schema).load('/content/drive/MyDrive/BigMess/NASA/NASA.csv')


# rozdzielenie kolumny Date na Year i Month
nasa.createOrReplaceTempView("nasa")
nasa = spark.sql("""
          SELECT
          CAST(SUBSTRING(CAST(Date AS STRING), 1, 4) AS INT) AS Year,
          CAST(SUBSTRING(CAST(Date AS STRING), 5, 2) AS INT) AS Month,
          n.*
          FROM nasa n
          """)

nasa = nasa.drop("Date")


"""
## Filtrowanie danych do scenariusza susz
"""

# filtrujemy nasza tabele przez wspolrzedne opisujace mniej wiecej tereny suszy w latach 1980-1985
arizona_coord = {'lon_min': -114.0587, 'lon_max': -109.0558, 'lat_min': 31.9, 'lat_max': 37.0}
teksas_coord= {'lon_min': -103.0651, 'lon_max': -93.5681, 'lat_min': 29.7628, 'lat_max': 34.135}


nasa_arizona_teksas = nasa_arizona_teksas = nasa.filter(
    ((col("lon") >= arizona_coord['lon_min']) &
     (col("lon") <= arizona_coord['lon_max']) &
     (col("lat") >= arizona_coord['lat_min']) &
     (col("lat") <= arizona_coord['lat_max'])) |
    ((col("lon") >= teksas_coord['lon_min']) &
     (col("lon") <= teksas_coord['lon_max']) &
     (col("lat") >= teksas_coord['lat_min']) &
     (col("lat") <= teksas_coord['lat_max']))
)


# susza w Teksasie i w Arizonie miala miejsce w latach 1980-1985
nasa_droughts = nasa_arizona_teksas.filter((col("Year") >= 1980) & (col("Year") <= 1985))


# policzymy ile punktow w danych zostalo po filtrowaniu
nasa_droughts.count()


nasa_droughts_pd = nasa_droughts.toPandas()


"""
## Filtrowanie danych do scenariusza powodzi
"""

# Huragan Harvey w 2017 spowodowal wiele powodzi w Teksasie
teksas_coord= {'lon_min': -103.0651, 'lon_max': -93.5681, 'lat_min': 29.7628, 'lat_max': 34.135}


nasa_teksas = nasa_arizona_teksas = nasa.filter(
    ((col("lon") >= teksas_coord['lon_min']) &
     (col("lon") <= teksas_coord['lon_max']) &
     (col("lat") >= teksas_coord['lat_min']) &
     (col("lat") <= teksas_coord['lat_max']))
)


# powodzie w Teksasie mialy miejsce w sierpniu i wrzesniu 2017 roku
nasa_floods = nasa_teksas.filter((col("Year") == 2017) & ((col("Month") >= 8) & (col("Month") <= 11)))


# policzymy ile punktow w danych zostalo po filtrowaniu
nasa_floods.count()


nasa_floods_pd = nasa_floods.toPandas()


"""
## Funkcje do rysowania map
"""

'''
Funkcja jako argumenty bierze listę wartości określających granice przedziałów liczbowych, które
będą określać jak dla rozważanego parametru mają zmieniać się kolory punktów, których lista stanowi
drugi argument funkcji.
'''
def get_colormap(values: list, colors_palette: list, name = 'custom'):
    values = np.sort(np.array(values))
    values = np.interp(values, (values.min(), values.max()), (0, 1))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(name, list(zip(values, colors_palette)))
    return cmap


def plot_map(df: pd.DataFrame, parameter_name: str, colormap: mpl.colors.LinearSegmentedColormap, title: str,
             point_size: int = 8, width: int = 800, height: int = 500, alpha: float = 1,
             bgcolor: str = 'white', colorbar_verbose: bool = True):

    gdf = gv.Points(df, ['lon', 'lat'], [parameter_name]) # obiekt zawierający punkty
    tiles = gts.OSM # wybór mapy tła, w tym wypadku OpenStreetMap

    # łączenie mapy tła z punktami i ustawienie wybranych parametrów wizualizacji
    map_with_points = tiles * gdf.opts(
        title=title,
        color=parameter_name,
        cmap=colormap,
        size=point_size,
        width=width,
        height=height,
        colorbar=colorbar_verbose,
        toolbar='above',
        tools=['hover', 'wheel_zoom', 'reset'],
        alpha=alpha, # przezroczystość
        bgcolor=bgcolor
    )

    return hv.render(map_with_points)


display(IFrame("https://www.google.com/maps/embed?pb=!1m14!1m12!1m3!1d13982681.959428234!2d-98.66341902257437!3d38.39997874427714!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!5e1!3m2!1spl!2spl!4v1703000232420!5m2!1spl!2spl", '800px', '500px'))


"""
## Charakterystyka omawianych terenów
* Arizona:
  * 4 pustynie, w zasadzie większa część Arizony to pustynia

* Teksas:
  * Jedynie zachodnia część Teksasu zalicza się do pustyni Chihuahua
"""

"""
## Algorytm k-srednich
"""

def MinMaxScaling(df: pd.DataFrame, attributes: list) -> pd.DataFrame:
  """
  Funkcja służąca do przeskalowania wybranych atrybutów za pomocą funkcji MinMaxScaler, a następnie stworzenia nowej ramki danych z tylko przeskalowanymi atrybutami.
  Parametry:
  - df (DataFrame): Pandas DataFrame zawierająca co najmniej atrybuty,
  - attributes (str): atrybuty, które będziemy skalować.
  """
  scaled_data = MinMaxScaler().fit_transform(df[attributes])
  scaled_df = pd.DataFrame(scaled_data, columns=attributes)
  return scaled_df


def do_kmeans_and_return_df_with_cluster_column(df: pd.DataFrame, scaled_df: pd.DataFrame, n_clusters: int, random_state: int) -> pd.DataFrame:
  """
  Funkcja wykonuje grupowanie k-średnich dla n_clusters klastrów oraz tworzy nową kolumnę z predykcjami algorytmu k-średnich w ramce danych df.
  Parametry:
  - df (DataFrame): Pandas DataFrame zawierająca co najmniej te same kolumny co scaled_df,
  - scaled_df (DataFrame): Pandas DataFrame zawierająca przeskalowane kolumny, na podstawie których dokonywane jest grupowanie,
  - n_clusters (int): maksymalna liczba klastrów,
  - random_state (int): ziarno losowości.
  """
  kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=random_state)
  kmeans.fit(scaled_df)
  pred = kmeans.predict(scaled_df)
  df['cluster'] = pred
  return df


"""
### Powodzie
"""

scaled_nasa_floods = MinMaxScaling(nasa_floods_pd, ['Rainf', 'Evap', 'AvgSurfT', 'Albedo', 'SoilT_40_100cm',  'GVEG', 'PotEvap', 'RootMoist','SoilM_100_200cm'])


output_floods = do_kmeans_and_return_df_with_cluster_column(nasa_floods_pd, scaled_nasa_floods, 2, 123)


colormap_cluster = get_colormap([0, max(output_floods.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_floods, parameter_name='cluster', colormap=colormap_cluster, title="K-srednich z obszarów objętych powodziami w roku 2017", alpha=1))


"""
Dla danych powodziowych nasz algorytm k-średnich zaklasyfikował cały obszar Teksasu jako niepystynia co nie jest do końca prawdą, niemniej jedank bardzo niewielka część Teksasu stanowi pustynie.
"""

"""
### Susze
"""

scaled_nasa_droughts = MinMaxScaling(nasa_droughts_pd, ['Rainf', 'Evap', 'AvgSurfT', 'Albedo', 'SoilT_40_100cm',  'GVEG', 'PotEvap', 'RootMoist','SoilM_100_200cm'])


output_droughts = do_kmeans_and_return_df_with_cluster_column(nasa_droughts_pd, scaled_nasa_droughts, 2, 123)


colormap_cluster = get_colormap([0, max(output_droughts.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_droughts, parameter_name='cluster', colormap=colormap_cluster, title="K-srednich z obszarów objetych suszami na przestzreni lat 1980-1985 ", alpha=0.9))


"""
Jeśli chodzi o dane suszowe to k-średnich pokazuje nam bardzo dziwne wyniki bo wciąż praktycznie wszystko klasyfikuje jako niepustynia co w przypadku Arizona nie ma zupełnie sensu, gdyż większość Arizony to obszary pustynne.
"""

"""
## Algorytm EM
"""

def do_EM_and_return_df_with_cluster_column(df: pd.DataFrame, scaled_df: pd.DataFrame, n_clusters: int, random_state: int) -> pd.DataFrame:
  """
  Funkcja wykonuje grupowani EM dla n_clusters klastrow oraz tworzy nową kolumnę z predykcjami algorytmu EM w ramce danych df.
  Parametry:
  - df (DataFrame): Pandas DataFrame zawierająca co najmniej te same kolumny co scaled_df,
  - scaled_df (DataFrame): Pandas DataFrame zawierająca przeskalowane kolumny, na podstawie których dokonywane jest grupowanie,
  - n_clusters (int): maksymalna liczba klastrow,
  - random_state (int): ziarno losowosci.
  """
  gm = GaussianMixture(n_components = n_clusters, n_init = 200, max_iter=200, init_params= 'random_from_data', covariance_type='spherical', random_state=random_state)
  gm_result = gm.fit_predict(scaled_df)
  new_df = df.copy()
  new_df['cluster'] = gm_result
  return new_df


"""
### Powodzie
"""

output_floods2 = do_EM_and_return_df_with_cluster_column(nasa_floods_pd, scaled_nasa_floods, 2, 123)


colormap_cluster = get_colormap([0, max(output_floods2.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_floods2, parameter_name='cluster', colormap=colormap_cluster, title="EM z obszarów objetych powodziami w 2017 roku", alpha=1))


"""
W przypadku EM dla danych powodziowych również uzyskujemy bardzo dziwne wyniki bo cały teren mamy zaklasyfikowany jako pustynie.
"""

"""
### Susze
"""

output_droughts2 = do_EM_and_return_df_with_cluster_column(nasa_droughts_pd, scaled_nasa_droughts, 2, 123)


colormap_cluster = get_colormap([0, max(output_droughts2.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_droughts2, parameter_name='cluster', colormap=colormap_cluster, title="EM z obszarow objetych  szuszami w latach 1980-1985", alpha=1))


"""
Jeśli chodzi o EM dla danych suszowych to tutaj znowu zupełnie nietrafne grupowanie. Większość terenów jako tereny niepustynne jedynie południowy-wschód Teksasu zaklasyfikowany jako pustynia co nie jest zgodne z prawdą.
"""

"""
## Drzewa decyzyjne
"""

# wczytujemy model i troche przerobimy tabele
model_m2_path='/content/drive/MyDrive/BigMess/NASA/Modele/Klasyfikacja/Stare/tree_classifier_m2'
with open(model_m2_path , 'rb') as f:
    model = pickle.load(f)

nasa_droughts_pd = nasa_droughts_pd.rename(columns = {'SoilT_40_100cm':'SoilT_10_40cm'})
nasa_floods_pd = nasa_floods_pd.rename(columns = {'SoilT_40_100cm':'SoilT_10_40cm'})

nasa_droughts_pd = nasa_droughts_pd.drop(columns = ['Month', 'Year', 'PotEvap', 'SoilM_100_200cm', 'cluster'])
nasa_floods_pd = nasa_floods_pd.drop(columns = ['Month', 'Year', 'PotEvap', 'SoilM_100_200cm', 'cluster'])


"""
### Susze
"""

tree_predict_droughts = model.predict(nasa_droughts_pd.iloc[:, ~nasa_droughts_pd.columns.isin(['lon', 'lat'])])
nasa_droughts_pd['Prediction'] = tree_predict_droughts


colormap_cluster = get_colormap([0, max(nasa_droughts_pd.Prediction.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=nasa_droughts_pd, parameter_name='Prediction', colormap=colormap_cluster, title="Drzewa decyzyjne z obszarów objetych suszami na przestzreni lat 1980-1985", alpha=0.5))


"""
W przypadku drzew decyzyjnych zastosowanych na danych o suszach dostaliśmy wyniki, które sugerują, że większość Arizony to niepustynia co nie jest prawdą. Większość Teksasu została zaklasyfikowana jako pystynia co też nie jest prawdą, ale ze względu na fakt, że Teksas był stanem, który najbardziej odczuł suszę w latach 80 uzyskane wyniki mają sens.
"""

"""
### Powodzie
"""

tree_predict_floods = model.predict(nasa_floods_pd.iloc[:, ~nasa_floods_pd.columns.isin(['lon', 'lat'])])
nasa_floods_pd['Prediction'] = tree_predict_floods


colormap_cluster = get_colormap([0, max(nasa_floods_pd.Prediction.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=nasa_floods_pd, parameter_name='Prediction', colormap=colormap_cluster, title="Drzewa decyzyjne z obszarów objętych powodziami w roku 2017", alpha=1))


"""
Klasyfikacja drzewami dla danych powodziowych nie dała sensownych rezultatów gdyż znowu większość Teksasu została zaklasyfikowana jako pustynia.
"""

