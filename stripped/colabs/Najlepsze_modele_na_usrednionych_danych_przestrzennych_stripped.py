!pip install pyspark
!pip install geoviews
!pip install datashader

import pyspark
import pickle
import pyspark.sql.functions as F
import pyspark.sql.types as T
import geoviews as gv
import colorcet as cc
import holoviews as hv
import datashader as ds
import matplotlib as mpl
import geoviews.tile_sources as gts
import datashader.transfer_functions as tf

from google.colab import drive
from pyspark.sql.functions import col
from pyspark.sql.functions import size
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType, StringType, StructType
from pyspark.sql import Window
from geopy.distance import geodesic
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from typing import Optional
from holoviews.operation.datashader import datashade
from holoviews import opts
from IPython.display import IFrame
from IPython.core.display import display
from bokeh.plotting import show, output_notebook

# tworzenie sesji w Sparku
spark = SparkSession.builder.appName('SparkWindows').getOrCreate()

# wczytanie danych z google drive
drive.mount('/content/drive')

columns = ['lon', 'lat', 'Date', 'Rainf', 'Evap', 'AvgSurfT', 'Albedo','SoilT_10_40cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilM_100_200cm']

# Utworzenie schematu okreslajacego typ zmiennych
schema = StructType()
for i in columns:
  if i == "Date":
    schema = schema.add(i, IntegerType(), True)
  else:
    schema = schema.add(i, FloatType(), True)

nasa = spark.read.format('csv').option("header", True).schema(schema).load('/content/drive/MyDrive/BigMess/NASA/NASA.csv')
nasa.createOrReplaceTempView("nasa")
nasa.show(5)


nasa_ym = spark.sql("""
          SELECT
          CAST(SUBSTRING(CAST(Date AS STRING), 1, 4) AS INT) AS Year,
          CAST(SUBSTRING(CAST(Date AS STRING), 5, 2) AS INT) AS Month,
          n.*
          FROM nasa n
          """)
nasa_ym = nasa_ym.drop("Date")

nasa_ym.createOrReplaceTempView("nasa_ym")

# wybieramy dane z lipca 2k23
SparkDataFrame_2023_7 = nasa_ym.filter('Year = 2023').filter('Month = 07')
SparkDataFrame_2023_7.show(5)

# konwertujemy do Pandasa
pd_2023_07 = SparkDataFrame_2023_7.toPandas()
pd_2023_07 = pd_2023_07.drop(columns = ['Year', 'Month'])

# wczytanie zbioru anotowanego
NASA_sample_an = pd.read_csv('/content/drive/MyDrive/BigMess/NASA/NASA_an.csv',sep=';')
NASA_sample_an['pustynia_i_step'] = NASA_sample_an.pustynia + NASA_sample_an.step
NASA_sample_an["pustynia_i_step"] = NASA_sample_an["pustynia_i_step"].apply(lambda x: 0 if x==1 else 1)

def create_lon_lat_pairs(df: pd.DataFrame, tolerance_lon: float = 5, tolerance_lat: float = 5, verbose: bool = False):
    projection = df[['lon', 'lat']]
    result = {}
    if verbose:
        count = 0
    for row in projection.itertuples(index=True):
        index, lon, lat = row.Index, row.lon, row.lat
        if verbose:
            count += 1
            print(f'Processing item no {count}')
        result[(lon, lat)] = set()
        for lon_other, lat_other in result:
            if (abs(lon - lon_other) < tolerance_lon) and (abs(lat - lat_other) < tolerance_lat):
                result[(lon, lat)].add((lon_other, lat_other))
                result[(lon_other, lat_other)].add((lon, lat))

    def generator():
        for key, values in result.items():
            lon, lat = key
            for value in values:
                lon_other, lat_other = value
                yield lon, lat, lon_other, lat_other

    columns = ['lon', 'lat', 'lon_other', 'lat_other']
    return pd.DataFrame(generator(), columns=columns)


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the Haversine distance between two points specified by their longitude and latitude.

    Parameters:
    lon1, lat1: Longitude and latitude of the first point
    lon2, lat2: Longitude and latitude of the second point

    Returns:
    Haversine distance in kilometers
    """
    R = 6371  # Earth radius in kilometers

    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Calculate distance
    distance = R * c
    return distance

def average_over_space_window(df: pd.DataFrame,max_distance: float, df_grid: Optional[pd.DataFrame] = None, tolerance_lon : float =1, tolerance_lat: float =1) -> pd.DataFrame:
  """
  some parameters works only id df_grid is left None
  df grid must also have proper column names
  """
  if df_grid is None:
    df_grid = create_lon_lat_pairs(df, tolerance_lon=tolerance_lon, tolerance_lat=tolerance_lat)
    df_grid['distance'] = haversine_distance(df_grid['lon'], df_grid['lat'], df_grid['lon_other'], df_grid['lat_other'])
  df_grid = df_grid[df_grid['distance'] < max_distance]
  df_grid = df_grid.drop(columns='distance')
  window_data = pd.merge(df_grid, df, left_on = ['lon_other', 'lat_other'], right_on = ['lon', 'lat'])
  window_data.rename(columns={'lon_x': 'lon', 'lat_x': 'lat'}, inplace=True)
  window_data.drop(['lon_other', 'lat_other', 'lon_y', 'lat_y'], axis=1, inplace=True)
  return window_data.groupby(['lon', 'lat']).mean().reset_index()



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

# tworzymy mozliwe pary wspolrzednych
df_grid = create_lon_lat_pairs(pd_2023_07, tolerance_lon=1, tolerance_lat=1, verbose = True)
df_grid.head()

# liczymy odleglosci
df_grid['distance'] = haversine_distance(df_grid['lon'], df_grid['lat'], df_grid['lon_other'], df_grid['lat_other'])
df_grid.sample(5)

df_grid[['lon', 'lat']].duplicated()

# wybieramy te ktore nie przekraczaja 50 jednostek
df_grid = df_grid[df_grid['distance'] < 50]

# zastepujemy wartosci pomiarowe wartosciami srednimi z okolicy punktu o srednicy 50km
result = average_over_space_window(pd_2023_07,max_distance=50, df_grid=df_grid)
result

scaled_pd_2023_7 = MinMaxScaling(result, ['Rainf','Evap' ,'AvgSurfT', 'Albedo', 'SoilT_10_40cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilM_100_200cm'])
output_pd_2023_7 = do_kmeans_and_return_df_with_cluster_column(result, scaled_pd_2023_7, 2, 123)

colormap_cluster = get_colormap([0, 1], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_pd_2023_7, parameter_name='cluster', colormap=colormap_cluster, title="K_średnich - lipiec 2023", alpha=1))

output_with_anotated = output_pd_2023_7.merge(NASA_sample_an, left_on=['lon','lat'], right_on = ['lon','lat'], how='inner')

positive = len(output_with_anotated.loc[output_with_anotated.cluster == output_with_anotated.pustynia_i_step])
accuracy = str(round(positive/len(NASA_sample_an)*100,2))
print("Accuracy na poziomie",accuracy+"%.")

# przetestujemy jak na naszych danych sprawdzi sie LightGBM
model_lgbm_m2_7_path='/content/drive/MyDrive/BigMess/NASA/Modele/Klasyfikacja/LightGBM/lgbm_m2_7'
with open(model_lgbm_m2_7_path , 'rb') as f:
    model_LGBM = pickle.load(f)

result = result.drop(columns = ['cluster'])
result_no_lonlat = result.drop(columns = ['lon', 'lat'])
LGBM_predict_2023_07 = model_LGBM.predict(result_no_lonlat)
result['Prediction'] = LGBM_predict_2023_07

colormap_cluster = get_colormap([0, max(result.Prediction.values)], ['darkgreen', 'yellow'])
output_notebook()
show(plot_map(df=result, parameter_name='Prediction', colormap=colormap_cluster, title="LightGBM - Lipiec 2023", alpha=0.5))

output_with_anotated = result.merge(NASA_sample_an, left_on=['lon','lat'], right_on = ['lon','lat'], how='inner')
positive = len(output_with_anotated.loc[output_with_anotated.Prediction == output_with_anotated.pustynia_i_step])
accuracy = str(round(positive/len(NASA_sample_an)*100,2))
print("Accuracy na poziomie",accuracy+"%.")

# teraz zobaczymy jak poradzą sobie lasy z naszymi danymi
model_las_m2_7_path='/content/drive/MyDrive/BigMess/NASA/Modele/Klasyfikacja/Lasy/las_m2_7'
with open(model_m3_path , 'rb') as f:
    model_forest = pickle.load(f)

result = result.drop(columns = ['AvgSurfT', 'Evap', 'PotEvap', 'Prediction'])
result_no_lonlat = result.drop(columns = ['lon', 'lat'])
forest_predict_2023_07 = model_forest.predict(result_no_lonlat)
result['Prediction'] = forest_predict_2023_07

colormap_cluster = get_colormap([0, max(result.Prediction.values)], ['darkgreen', 'yellow'])
output_notebook()
show(plot_map(df=result, parameter_name='Prediction', colormap=colormap_cluster, title="Lasy - Lipiec 2023", alpha=0.5))

output_EM = do_EM_and_return_df_with_cluster_column(result, scaled_pd_2023_7, 2, 123)

colormap_cluster = get_colormap([0, max(output_EM.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_EM, parameter_name='cluster', colormap=colormap_cluster, title="EM - Lipiec 2023", alpha=1))

output_with_anotated = output_EM.merge(NASA_sample_an, left_on=['lon','lat'], right_on = ['lon','lat'], how='inner')
positive = len(output_with_anotated.loc[output_with_anotated.cluster == output_with_anotated.pustynia_i_step])
accuracy = str(round(positive/len(NASA_sample_an)*100,2))
print("Accuracy na poziomie",accuracy+"%.")
