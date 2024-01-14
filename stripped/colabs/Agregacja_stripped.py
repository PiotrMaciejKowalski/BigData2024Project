!pip install datashader
!pip install holoviews hvplot colorcet
!pip install geoviews

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import copy
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib as mpl
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
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, jaccard_score, recall_score
import pickle
import seaborn as sns
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from typing import Optional, Tuple, List
from imblearn.over_sampling import RandomOverSampler, SMOTE
from bokeh.layouts import row

from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/PiotrMaciejKowalski/BigData2024Project.git

!chmod 755 /content/BigData2024Project/src/setup.sh
!/content/BigData2024Project/src/setup.sh

import sys
sys.path.append('/content/BigData2024Project/src')

from start_spark import initialize_spark
initialize_spark()

from big_mess.loaders import default_loader
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame, functions as F

spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()

%%time
nasa = default_loader(spark)
# nasa.show(5)

nasa.createOrReplaceTempView("nasa")

from big_mess.loaders import load_anotated
nasa_anotated = load_anotated(spark)

nasa_anotated.createOrReplaceTempView("nasa_anotated")

def nasa_agg_year_with_avg(sdf: SparkDataFrame, year: int) -> SparkDataFrame:
    """
    Funkcja sluzaca do ograniczenia zbioru NASA do jednego roku,a następnie wykonująca agregacje roczna:
    suma miesieczna -> suma roczna (Rainf, Evap)
    srednia miesieczna -> srednia roczna (pozostale zmienne)
    """
    data = sdf.filter(f'Year = {year}')
    data.createOrReplaceTempView("data")
    data_group_by = spark.sql("""
          SELECT
          lon, lat,
          SUM(Rainf) AS Rainf,
          SUM(Evap) AS Evap,
          AVG(AvgSurfT) AS AvgSurfT,
          AVG(Albedo) AS Albedo,
          AVG(SoilT_40_100cm) AS SoilT_40_100cm,
          AVG(GVEG) AS GVEG,
          AVG(PotEvap)As PotEvap,
          AVG(RootMoist) AS RootMoist,
          AVG(SoilM_100_200cm) AS SoilM_100_200cm
          FROM data
          GROUP BY lon, lat
          ORDER BY lon, lat
          """)
    return data_group_by

def nasa_agg_year_with_median(sdf: SparkDataFrame, year: int) -> SparkDataFrame:
    """
    Funkcja sluzaca do ograniczenia zbioru NASA do jednego roku,a następnie wykonująca agregacje roczna:
    suma miesieczna -> suma roczna (Rainf, Evap)
    srednia miesieczna -> mediana (pozostale zmienne)
    """
    data = sdf.filter(f'Year = {year}')
    data.createOrReplaceTempView("data")
    data_group_by = spark.sql("""
          SELECT
          lon, lat,
          SUM(Rainf) AS Rainf,
          SUM(Evap) AS Evap,
          MEDIAN(AvgSurfT) AS AvgSurfT,
          MEDIAN(Albedo) AS Albedo,
          MEDIAN(SoilT_40_100cm) AS SoilT_40_100cm,
          MEDIAN(GVEG) AS GVEG,
          MEDIAN(PotEvap)As PotEvap,
          MEDIAN(RootMoist) AS RootMoist,
          MEDIAN(SoilM_100_200cm) AS SoilM_100_200cm
          FROM data
          GROUP BY lon, lat
          ORDER BY lon, lat
          """)
    return data_group_by

nasa_2020_avg = nasa_agg_year_with_avg(nasa, 2020)
df_2020_avg = nasa_2020_avg.toPandas()

nasa_2020_median = nasa_agg_year_with_median(nasa, 2020)
df_2020_median = nasa_2020_median.toPandas()

nasa_2000_avg = nasa_agg_year_with_avg(nasa, 2000)
df_2000_avg = nasa_2000_avg.toPandas()

nasa_2000_median = nasa_agg_year_with_median(nasa, 2000)
df_2000_median = nasa_2000_median.toPandas()

nasa_1979_avg = nasa_agg_year_with_avg(nasa, 1979)
df_1979_avg = nasa_1979_avg.toPandas()

nasa_1979_median = nasa_agg_year_with_median(nasa, 1979)
df_1979_median = nasa_1979_median.toPandas()

# funkcja do normalizacji danych z notatnika Mariusza
def MinMaxScaling(df: pd.DataFrame, attributes: list) -> pd.DataFrame:
  """
  Funkcja sluzaca do przeskalowania wybranych atrybutow za pomoca funkcji MinMaxScaler, a nastepnie stworzenia nowej ramki danych z tylko przeskalowanymi atrybutami.
  Parametry:
  - df (DataFrame): Pandas DataFrame zawierajaca co najmniej atrybuty,
  - attributes (str): atrybuty, ktore bedziemy skalowac.
  """
  scaled_data = MinMaxScaler().fit_transform(df[attributes])
  scaled_df = pd.DataFrame(scaled_data, columns=attributes)
  return scaled_df

# normalizacja danych z notatnika Izy - przerobione na funkcje analogiczna do MinMaxScaling
def standarized(df: pd.DataFrame, attributes: list) -> pd.DataFrame:
  """
  Funkcja sluzaca do przeskalowania wybranych atrybutow za pomoca funkcji StandardScaler, a nastepnie stworzenia nowej ramki danych z tylko przeskalowanymi atrybutami.
  Parametry:
  - df (DataFrame): Pandas DataFrame zawierajaca co najmniej atrybuty,
  - attributes (str): atrybuty, które bedziemy skalować.
  """
  scaler = preprocessing.StandardScaler()
  scaled_data = scaler.fit_transform(df[attributes])
  scaled_df = pd.DataFrame(scaled_data, columns=attributes)
  return scaled_df

# funkcja do grupowania algorytymem k-srednich z notatnika Mariusza (zwraca new_df zamiast oryginalnie df)
def do_kmeans_and_return_df_with_cluster_column(df: pd.DataFrame, scaled_df: pd.DataFrame, n_clusters: int, random_state: int) -> pd.DataFrame:
  """
  Funkcja wykonuje grupowanie k-srednich dla n_clusters klastrow oraz tworzy nową kolumne z predykcjami algorytmu k-srednich w ramce danych df.
  Parametry:
  - df (DataFrame): Pandas DataFrame zawierajaca co najmniej te same kolumny co scaled_df,
  - scaled_df (DataFrame): Pandas DataFrame zawierajaca przeskalowane kolumny, na podstawie ktorych dokonywane jest grupowanie,
  - n_clusters (int): maksymalna liczba klastrow,
  - random_state (int): ziarno losowosci.
  """
  kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=random_state)
  kmeans.fit(scaled_df)
  pred = kmeans.predict(scaled_df)
  new_df = df.copy()
  new_df['cluster'] = pred
  return new_df

# analagoczna funkcja do do_kmeans_and_return_df_with_cluster_column dla grupowania EM
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

# funkcja do rysowania map z notatnika Mariusza
def get_colormap(values: list, colors_palette: list, name = 'custom'):
    """
    Funkcja jako argumenty bierze liste wartosci okreslajacych granice przedzialow liczbowych, ktore
    beda okreslac jak dla rozwazanego parametru maja zmieniac się kolory punktow, ktorych lista stanowi
    drugi argument funkcji.
    """
    values = np.sort(np.array(values))
    values = np.interp(values, (values.min(), values.max()), (0, 1))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(name, list(zip(values, colors_palette)))
    return cmap

# funkcja do rysowania map z notatnika Mariusza
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

# funkcja do zbalansawonia zbioru danych z notatnika Rafala
class BalanceDataSet():
  '''
  Two techniques for handling imbalanced data.
  '''
  def __init__(
      self,
      X: DataFrame,
      y: DataFrame
      ) -> None:
      self.X = X
      self.y = y
      assert len(self.X)==len(self.y)

  def useOverSampling(
      self,
      randon_seed: Optional[int] = 2023
      ) -> Tuple[DataFrame, DataFrame]:
    oversample = RandomOverSampler( sampling_strategy='auto',
                  random_state=randon_seed)
    return oversample.fit_resample(self.X, self.y)

  def useSMOTE(
      self,
      randon_seed: Optional[int] = 2023
      ) -> Tuple[DataFrame, DataFrame]:
    smote = SMOTE(random_state=randon_seed)
    return smote.fit_resample(self.X, self.y)

# funkcja do podsumowania modelu z notatnika Rafala
def summary_model(model, X:pd.DataFrame, y:pd.DataFrame, labels_names: List) -> None:
  y_pred = model.predict(X)
  y_real= y
  cf_matrix = confusion_matrix(y_real, y_pred)
  group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
  group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
  labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
  labels = np.asarray(labels).reshape(len(labels_names),len(labels_names))
  sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Reds',xticklabels=labels_names,yticklabels=labels_names)
  plt.xlabel('Predykcja')
  plt.ylabel('Rzeczywistość')
  plt.show()

display(IFrame("https://www.google.com/maps/embed?pb=!1m14!1m12!1m3!1d13982681.959428234!2d-98.66341902257437!3d38.39997874427714!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!5e1!3m2!1spl!2spl!4v1703000232420!5m2!1spl!2spl", '800px', '500px'))

df_anotated = nasa_anotated.toPandas()

colormap_cluster = get_colormap([0, max(df_anotated.pustynia.values)], ['yellow', 'darkgreen'])
plot_anotated = plot_map(df=df_anotated, parameter_name='pustynia', colormap=colormap_cluster, title="Dataset anotowany: 0 - niepustynia, 1 - pustynia", alpha=1)
output_notebook()
show(plot_anotated)

col = ['Rainf', 'Evap',  'AvgSurfT', 'Albedo', 'SoilT_40_100cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilM_100_200cm']

scaled_df_2020_median_stand = standarized(df_2020_median, col)
scaled_df_2020_median_MinMax = MinMaxScaling(df_2020_median, col)
scaled_df_2020_avg_stand = standarized(df_2020_avg, col)
scaled_df_2020_avg_MinMax = MinMaxScaling(df_2020_avg, col)

scaled_df_2000_median_stand = standarized(df_2000_median, col)
scaled_df_2000_median_MinMax = MinMaxScaling(df_2000_median, col)
scaled_df_2000_avg_stand = standarized(df_2000_avg, col)
scaled_df_2000_avg_MinMax = MinMaxScaling(df_2000_avg, col)

scaled_df_1979_median_stand = standarized(df_1979_median, col)
scaled_df_1979_median_MinMax = MinMaxScaling(df_1979_median, col)
scaled_df_1979_avg_stand = standarized(df_1979_avg, col)
scaled_df_1979_avg_MinMax = MinMaxScaling(df_1979_avg, col)

output_df_2020_median = do_kmeans_and_return_df_with_cluster_column(df_2020_median, scaled_df_2020_median_stand, 2, 123)
colormap_cluster = get_colormap([0, max(output_df_2020_median.cluster.values)], ['yellow', 'darkgreen'])
plot_1 = plot_map(df=output_df_2020_median, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 2020 (median + standarized)", alpha=0.5)

output_df_2020_median = do_kmeans_and_return_df_with_cluster_column(df_2020_median, scaled_df_2020_median_MinMax, 2, 123)
colormap_cluster = get_colormap([0, max(output_df_2020_median.cluster.values)], ['yellow', 'darkgreen'])
plot_2 = plot_map(df=output_df_2020_median, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 2020 (median + MinMaxScaling)", alpha=0.5)

output_notebook()
show(row(plot_1, plot_2))

output_df_2020_avg = do_kmeans_and_return_df_with_cluster_column(df_2020_avg, scaled_df_2020_avg_stand, 2, 123)
colormap_cluster = get_colormap([0, max(output_df_2020_avg.cluster.values)], ['yellow', 'darkgreen'])
plot_1 = plot_map(df=output_df_2020_avg, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 2020 (avg + standarized)", alpha=0.5)

output_df_2020_avg = do_kmeans_and_return_df_with_cluster_column(df_2020_avg, scaled_df_2020_avg_MinMax, 2, 123)
colormap_cluster = get_colormap([0, max(output_df_2020_avg.cluster.values)], ['yellow', 'darkgreen'])
plot_2 = plot_map(df=output_df_2020_avg, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 2020 (avg + MinMaxScaling)", alpha=0.5)

output_notebook()
show(row(plot_1, plot_2))

output_df_2000_median = do_kmeans_and_return_df_with_cluster_column(df_2000_median, scaled_df_2000_median_stand, 2, 123)
colormap_cluster = get_colormap([0, max(output_df_2000_median.cluster.values)], ['yellow', 'darkgreen'])
plot_1 = plot_map(df=output_df_2000_median, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 2000 (median + standarized)", alpha=0.5)

output_df_2000_median = do_kmeans_and_return_df_with_cluster_column(df_2000_median, scaled_df_2000_median_MinMax, 2, 123)
colormap_cluster = get_colormap([0, max(output_df_2000_median.cluster.values)], ['yellow', 'darkgreen'])
plot_2 = plot_map(df=output_df_2000_median, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 2000 (median + MinMaxScaling)", alpha=0.5)

output_notebook()
show(row(plot_1, plot_2))

output_df_2000_avg = do_kmeans_and_return_df_with_cluster_column(df_2000_avg, scaled_df_2000_avg_stand, 2, 123)
colormap_cluster = get_colormap([0, max(output_df_2000_avg.cluster.values)], ['yellow', 'darkgreen'])
plot_1 = plot_map(df=output_df_2000_avg, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 2000 (avg + standarized)", alpha=0.5)

output_df_2000_avg = do_kmeans_and_return_df_with_cluster_column(df_2000_avg, scaled_df_2000_avg_MinMax, 2, 123)
colormap_cluster = get_colormap([0, max(output_df_2000_avg.cluster.values)], ['yellow', 'darkgreen'])
plot_2 = plot_map(df=output_df_2000_avg, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 2000 (avg + MinMaxScaling)", alpha=0.5)

output_notebook()
show(row(plot_1, plot_2))

output_df_1979_median = do_kmeans_and_return_df_with_cluster_column(df_1979_median, scaled_df_1979_median_stand, 2, 123)
colormap_cluster = get_colormap([0, max(output_df_1979_median.cluster.values)], ['yellow', 'darkgreen'])
plot_1 = plot_map(df=output_df_1979_median, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 1979 (median + standarized)", alpha=0.5)

output_df_1979_median = do_kmeans_and_return_df_with_cluster_column(df_1979_median, scaled_df_1979_median_MinMax, 2, 123)
plot_2 = plot_map(df=output_df_1979_median, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 1979 (median + MinMaxScaling)", alpha=0.5)

output_notebook()
show(row(plot_1, plot_2))

output_df_1979_avg = do_kmeans_and_return_df_with_cluster_column(df_1979_avg, scaled_df_1979_avg_stand, 2, 123)
colormap_cluster = get_colormap([0, max(output_df_1979_avg.cluster.values)], ['yellow', 'darkgreen'])
plot_1 = plot_map(df=output_df_1979_avg, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 1979 (avg + standarized)", alpha=0.5)

output_df_1979_avg = do_kmeans_and_return_df_with_cluster_column(df_1979_avg, scaled_df_1979_avg_MinMax, 2, 123)
colormap_cluster = get_colormap([0, max(output_df_1979_avg.cluster.values)], ['yellow', 'darkgreen'])
plot_2 = plot_map(df=output_df_1979_avg, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 1979 (avg + MinMaxScaling)", alpha=0.5)

output_notebook()
show(row(plot_1, plot_2))

output_df_2020_EM_median = do_EM_and_return_df_with_cluster_column(df_2020_median, scaled_df_2020_median_stand, 2, 177)
colormap_cluster = get_colormap([0, max(output_df_2020_EM_median.cluster.values)], ['yellow', 'darkgreen'])
plot_1 = plot_map(df=output_df_2020_EM_median, parameter_name='cluster', colormap=colormap_cluster, title="EM dla danych z 2020 (median + standarized)", alpha=0.5)

output_df_2020_EM_median = do_EM_and_return_df_with_cluster_column(df_2020_median, scaled_df_2020_median_MinMax, 2, 177)
colormap_cluster = get_colormap([0, max(output_df_2020_EM_median.cluster.values)], ['yellow', 'darkgreen'])
plot_2 = plot_map(df=output_df_2020_EM_median, parameter_name='cluster', colormap=colormap_cluster, title="EM dla danych z 2020 (median + MinMaxScaling)", alpha=0.5)

output_notebook()
show(row(plot_1, plot_2))

output_df_2020_EM_avg = do_EM_and_return_df_with_cluster_column(df_2020_avg, scaled_df_2020_avg_stand, 2, 177)
colormap_cluster = get_colormap([0, max(output_df_2020_EM_avg.cluster.values)], ['yellow', 'darkgreen'])
plot_1 = plot_map(df=output_df_2020_EM_avg, parameter_name='cluster', colormap=colormap_cluster, title="EM dla danych z 2020 (avg + standarized)", alpha=0.5)

output_df_2020_EM_avg = do_EM_and_return_df_with_cluster_column(df_2020_avg, scaled_df_2020_avg_MinMax, 2, 177)
colormap_cluster = get_colormap([0, max(output_df_2020_EM_avg.cluster.values)], ['yellow', 'darkgreen'])
plot_2 = plot_map(df=output_df_2020_EM_avg, parameter_name='cluster', colormap=colormap_cluster, title="EM dla danych z 2020 (avg + MinMaxScaling)", alpha=0.5)

output_notebook()
show(row(plot_1, plot_2))

from big_mess.heuristic_classifier import heuristic_classify

heuristic_2020_avg = heuristic_classify(nasa_2020_avg)
df_heuristic_2020_avg = heuristic_2020_avg.toPandas()

colormap_cluster = get_colormap([0, max(df_heuristic_2020_avg.Pustynia.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=df_heuristic_2020_avg, parameter_name='Pustynia', colormap=colormap_cluster, title="Klasyfikacja heurystyczna dla danych z 2020 (avg)", alpha=0.5))

nasa_2020_avg.createOrReplaceTempView("nasa_2020_avg")
nasa_2020_avg_all = spark.sql("""
          SELECT
          lon, lat,
          Rainf/12 as Rainf,
          Evap/12 as Evap,
          AvgSurfT,
          Albedo,
          SoilT_40_100cm,
          GVEG,
          PotEvap,
          RootMoist,
          SoilM_100_200cm
          FROM nasa_2020_avg
          """)

heuristic_2020_avg_all = heuristic_classify(nasa_2020_avg_all)
df_heuristic_2020_avg_all = heuristic_2020_avg_all.toPandas()

colormap_cluster = get_colormap([0, max(df_heuristic_2020_avg_all.Pustynia.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=df_heuristic_2020_avg_all, parameter_name='Pustynia', colormap=colormap_cluster, title="Klasyfikacja heurystyczna dla danych z 2020 (avg all)", alpha=0.5))

heuristic_2020_median = heuristic_classify(nasa_2020_median)
df_heuristic_2020_median = heuristic_2020_median.toPandas()

colormap_cluster = get_colormap([0, max(df_heuristic_2020_median.Pustynia.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=df_heuristic_2020_median, parameter_name='Pustynia', colormap=colormap_cluster, title="Klasyfikacja heurystyczna dla danych z 2020 (median)", alpha=0.5))

nasa_2020_median_all = spark.sql("""
          SELECT
          lon, lat,
          MEDIAN(Rainf) AS Rainf,
          MEDIAN(Evap) AS Evap,
          MEDIAN(AvgSurfT) AS AvgSurfT,
          MEDIAN(Albedo) AS Albedo,
          MEDIAN(SoilT_40_100cm) AS SoilT_40_100cm,
          MEDIAN(GVEG) AS GVEG,
          MEDIAN(PotEvap)As PotEvap,
          MEDIAN(RootMoist) AS RootMoist,
          MEDIAN(SoilM_100_200cm) AS SoilM_100_200cm
          FROM nasa
          WHERE Year = 2020
          GROUP BY lon, lat
          """)

heuristic_2020_median_all = heuristic_classify(nasa_2020_median_all)
df_heuristic_2020_median_all = heuristic_2020_median_all.toPandas()

colormap_cluster = get_colormap([0, max(df_heuristic_2020_median_all.Pustynia.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=df_heuristic_2020_median_all, parameter_name='Pustynia', colormap=colormap_cluster, title="Klasyfikacja heurystyczna dla danych z 2020 (median all)", alpha=0.5))

nasa_2020_avg_anotated = spark.sql("""
          SELECT
          n.lon, n.lat,
          n.Rainf,
          n.Evap,
          n.AvgSurfT,
          n.Albedo,
          n.SoilT_40_100cm,
          n.GVEG,
          n.PotEvap,
          n.RootMoist,
          n.SoilM_100_200cm,
          na.pustynia
          FROM nasa_2020_avg n
          LEFT JOIN nasa_anotated na ON na.lon = n.lon AND na.lat = n.lat
          WHERE na.pustynia is not null
          """)

df_2020_avg_anotated = nasa_2020_avg_anotated.toPandas()

X = df_2020_avg_anotated.loc[:,'Rainf':'SoilM_100_200cm']
y = df_2020_avg_anotated['pustynia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)
X_train_bal, y_train_bal = BalanceDataSet(X_train, y_train).useSMOTE()

tree_classifier_m1 = tree.DecisionTreeClassifier(random_state = 2023)

tree_classifier_m1.fit(X_train_bal, y_train_bal)

print("classifier accuracy {:.2f}%".format(tree_classifier_m1.score(X_test,  y_test) * 100))

summary_model(tree_classifier_m1, X_test, y_test, ['0', '1'])

tree_classifier_m1_pred = tree_classifier_m1.predict(df_2020_avg.loc[:,'Rainf':'SoilM_100_200cm'])

df_2020_avg_predict = df_2020_avg.copy()
df_2020_avg_predict['Pustynia'] = tree_classifier_m1_pred

colormap_cluster = get_colormap([0, max(df_2020_avg_predict.Pustynia.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=df_2020_avg_predict, parameter_name='Pustynia', colormap=colormap_cluster, title="Drzewa decyzyjne dla danych z 2020 (avg)", alpha=0.5))

nasa_2020_median.createOrReplaceTempView("nasa_2020_median")
nasa_2020_median_anotated = spark.sql("""
          SELECT
          n.lon, n.lat,
          n.Rainf,
          n.Evap,
          n.AvgSurfT,
          n.Albedo,
          n.SoilT_40_100cm,
          n.GVEG,
          n.PotEvap,
          n.RootMoist,
          n.SoilM_100_200cm,
          na.pustynia
          FROM nasa_2020_median n
          LEFT JOIN nasa_anotated na ON na.lon = n.lon AND na.lat = n.lat
          WHERE na.pustynia is not null
          """)

df_2020_median_anotated = nasa_2020_median_anotated.toPandas()

X_2 = df_2020_median_anotated.loc[:,'Rainf':'SoilM_100_200cm']
y_2 = df_2020_median_anotated['pustynia']

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=2023)
X_train_bal_2, y_train_bal_2 = BalanceDataSet(X_train_2, y_train_2).useSMOTE()

tree_classifier_m2 = tree.DecisionTreeClassifier(random_state = 2023)

tree_classifier_m2.fit(X_train_bal_2, y_train_bal_2)

print("classifier accuracy {:.2f}%".format(tree_classifier_m2.score(X_test_2,  y_test_2) * 100))

summary_model(tree_classifier_m2, X_test_2, y_test_2, ['0', '1'])

tree_classifier_m2_pred = tree_classifier_m2.predict(df_2020_median.loc[:,'Rainf':'SoilM_100_200cm'])

df_2020_median_predict = df_2020_median.copy()
df_2020_median_predict['Pustynia'] = tree_classifier_m2_pred

colormap_cluster = get_colormap([0, max(df_2020_median_predict.Pustynia.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=df_2020_median_predict, parameter_name='Pustynia', colormap=colormap_cluster, title="Drzewa decyzyjne dla danych z 2020 (median)", alpha=0.5))

nasa_2000_avg.createOrReplaceTempView("nasa_2000_avg")
nasa_2000_avg_anotated = spark.sql("""
          SELECT
          n.lon, n.lat,
          n.Rainf,
          n.Evap,
          n.AvgSurfT,
          n.Albedo,
          n.SoilT_40_100cm,
          n.GVEG,
          n.PotEvap,
          n.RootMoist,
          n.SoilM_100_200cm,
          na.pustynia
          FROM nasa_2000_avg n
          LEFT JOIN nasa_anotated na ON na.lon = n.lon AND na.lat = n.lat
          WHERE na.pustynia is not null
          """)

df_2000_avg_anotated = nasa_2000_avg_anotated.toPandas()

X_3 = df_2000_avg_anotated.loc[:,'Rainf':'SoilM_100_200cm']
y_3 = df_2000_avg_anotated['pustynia']

X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3, y_3, test_size=0.2, random_state=2023)
X_train_bal_3, y_train_bal_3 = BalanceDataSet(X_train, y_train).useSMOTE()

tree_classifier_m3 = tree.DecisionTreeClassifier(random_state = 2023)

tree_classifier_m3.fit(X_train_bal_3, y_train_bal_3)

print("classifier accuracy {:.2f}%".format(tree_classifier_m3.score(X_test_3,  y_test_3) * 100))

summary_model(tree_classifier_m1, X_test_3, y_test_3, ['0', '1'])

tree_classifier_m3_pred = tree_classifier_m3.predict(df_2000_avg.loc[:,'Rainf':'SoilM_100_200cm'])

df_2000_avg_predict = df_2000_avg.copy()
df_2000_avg_predict['Pustynia'] = tree_classifier_m3_pred

colormap_cluster = get_colormap([0, max(df_2000_avg_predict.Pustynia.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=df_2000_avg_predict, parameter_name='Pustynia', colormap=colormap_cluster, title="Drzewa decyzyjne dla danych z 2000 (avg)", alpha=0.5))
