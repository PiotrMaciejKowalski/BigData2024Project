!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q dlcdn.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
!tar xf spark-3.5.0-bin-hadoop3.tgz
!pip install -q findspark

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.5.0-bin-hadoop3"

import findspark
findspark.init()

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from google.colab import drive
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.types import IntegerType, FloatType, StringType, StructType
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import copy
import warnings
warnings.filterwarnings('ignore')

spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()

drive.mount('/content/drive')

columns = ['lon', 'lat', 'Date', 'Rainf', 'Evap', 'AvgSurfT', 'Albedo','SoilT_10_40cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilM_100_200cm']

# Utworzenie schematu określającego typ zmiennych
schema = StructType()
for i in columns:
  if i == "Date":
    schema = schema.add(i, IntegerType(), True)
  else:
    schema = schema.add(i, FloatType(), True)

# Wczytanie zbioru Nasa w sparku
nasa = spark.read.format('csv').option("header", True).schema(schema).load('/content/drive/MyDrive/BigMess/NASA/NASA.csv')

nasa.createOrReplaceTempView("nasa")

nasa_ym = spark.sql("""
          SELECT
          CAST(SUBSTRING(CAST(Date AS STRING), 1, 4) AS INT) AS Year,
          CAST(SUBSTRING(CAST(Date AS STRING), 5, 2) AS INT) AS Month,
          n.*
          FROM nasa n
          """)
nasa_ym = nasa_ym.drop("Date")

nasa_ym.createOrReplaceTempView("nasa_ym")

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

SparkDataFrame_2023_7 = spark.sql("""
                        SELECT
                        *
                        FROM nasa_ym WHERE (Year == 2023) and (Month == 7)
                        order by lon, lat, Year, Month
                        """)

SparkDataFrame_2023_7.show(5)

df_2023_7 = SparkDataFrame_2023_7.toPandas()

df_2023_7.describe()

scaled_df_2023_7 = MinMaxScaling(df_2023_7, ['Evap', 'Rainf', 'AvgSurfT', 'Albedo', 'SoilM_100_200cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilT_10_40cm'])

output_df_2023_7 = do_kmeans_and_return_df_with_cluster_column(df_2023_7, scaled_df_2023_7, 2, 123)

SparkDataFrame_2023_7_pca = spark.sql("""
                        SELECT
                        *
                        FROM nasa_ym WHERE (Year == 2023) and (Month == 7)
                        order by lon, lat, Year, Month
                        """)

df_2023_7_pca = SparkDataFrame_2023_7_pca.toPandas()

# Standaryzacja
standarized_df_2023_7_pca = StandardScaler().fit_transform(df_2023_7_pca.drop(columns=["Month", "lon", "lat", "Year"]))
# PCA
pca = PCA(n_components=2)
standarized_df_2023_7_pca_done = pca.fit_transform(standarized_df_2023_7_pca)

# Wyjaśniona wariancja
pca.explained_variance_ratio_

standarized_df_2023_7_pca_done = pd.DataFrame(data = standarized_df_2023_7_pca_done, columns = ['principal component 1', 'principal component 2'])

output_df_2023_7_pca = do_kmeans_and_return_df_with_cluster_column(df_2023_7_pca, standarized_df_2023_7_pca_done, 2, 123)

SparkDataFrame_2023_1 = spark.sql("""
                        SELECT
                        *
                        FROM nasa_ym WHERE (Year == 2023) and (Month == 1)
                        order by lon, lat, Year, Month
                        """)

df_2023_1 = SparkDataFrame_2023_1.toPandas()

scaled_df_2023_1 = MinMaxScaling(df_2023_1, ['Evap', 'Rainf', 'AvgSurfT', 'Albedo', 'SoilM_100_200cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilT_10_40cm'])

output_df_2023_1 = do_kmeans_and_return_df_with_cluster_column(df_2023_1, scaled_df_2023_1, 2, 123)

SparkDataFrame_2023_1_pca = spark.sql("""
                        SELECT
                        *
                        FROM nasa_ym WHERE (Year == 2023) and (Month == 1)
                        order by lon, lat, Year, Month
                        """)

df_2023_1_pca = SparkDataFrame_2023_1_pca.toPandas()

standarized_df_2023_1_pca = StandardScaler().fit_transform(df_2023_1_pca.drop(columns=["Month", "lon", "lat", "Year"]))

pca = PCA(n_components=2)
standarized_df_2023_1_pca_done = pca.fit_transform(standarized_df_2023_1_pca)

pca.explained_variance_ratio_

standarized_df_2023_1_pca_done = pd.DataFrame(data = standarized_df_2023_1_pca_done, columns = ['principal component 1', 'principal component 2'])

output_df_2023_1_pca = do_kmeans_and_return_df_with_cluster_column(df_2023_1_pca, standarized_df_2023_1_pca_done, 2, 123)

SparkDataFrame_2000_7 = spark.sql("""
                        SELECT
                        *
                        FROM nasa_ym WHERE (Year == 2000) and (Month == 7)
                        order by lon, lat, Year, Month
                        """)

df_2000_7 = SparkDataFrame_2000_7.toPandas()

scaled_df_2000_7 = MinMaxScaling(df_2000_7, ['Evap', 'Rainf', 'AvgSurfT', 'Albedo', 'SoilM_100_200cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilT_10_40cm'])

output_df_2000_7 = do_kmeans_and_return_df_with_cluster_column(df_2000_7, scaled_df_2000_7, 2, 123)

SparkDataFrame_2000_7_pca = spark.sql("""
                        SELECT
                        *
                        FROM nasa_ym WHERE (Year == 2000) and (Month == 7)
                        order by lon, lat, Year, Month
                        """)

df_2000_7_pca = SparkDataFrame_2000_7_pca.toPandas()

# Standaryzacja
standarized_df_2000_7_pca = StandardScaler().fit_transform(df_2000_7_pca.drop(columns=["Month", "lon", "lat", "Year"]))
# PCA
pca = PCA(n_components=3)
standarized_df_2000_7_pca_done = pca.fit_transform(standarized_df_2000_7_pca)

# Wyjaśniona wariancja
pca.explained_variance_ratio_

standarized_df_2000_7_pca_done = pd.DataFrame(data = standarized_df_2000_7_pca_done, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

output_df_2000_7_pca = do_kmeans_and_return_df_with_cluster_column(df_2000_7_pca, standarized_df_2000_7_pca_done, 2, 123)

SparkDataFrame_2000_1 = spark.sql("""
                        SELECT
                        *
                        FROM nasa_ym WHERE (Year == 2000) and (Month == 1)
                        order by lon, lat, Year, Month
                        """)

df_2000_1 = SparkDataFrame_2000_1.toPandas()

scaled_df_2000_1 = MinMaxScaling(df_2000_1, ['Evap', 'Rainf', 'AvgSurfT', 'Albedo', 'SoilM_100_200cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilT_10_40cm'])

output_df_2000_1 = do_kmeans_and_return_df_with_cluster_column(df_2000_1, scaled_df_2000_1, 2, 123)

SparkDataFrame_2000_1_pca = spark.sql("""
                        SELECT
                        *
                        FROM nasa_ym WHERE (Year == 2000) and (Month == 1)
                        order by lon, lat, Year, Month
                        """)

df_2000_1_pca = SparkDataFrame_2000_1_pca.toPandas()

standarized_df_2000_1_pca = StandardScaler().fit_transform(df_2000_1_pca.drop(columns=["Month", "lon", "lat", "Year"]))

pca = PCA(n_components=2)
standarized_df_2000_1_pca_done = pca.fit_transform(standarized_df_2000_1_pca)

pca.explained_variance_ratio_

standarized_df_2000_1_pca_done = pd.DataFrame(data = standarized_df_2000_1_pca_done, columns = ['principal component 1', 'principal component 2'])

output_df_2000_1_pca = do_kmeans_and_return_df_with_cluster_column(df_2000_1_pca, standarized_df_2000_1_pca_done, 2, 123)

SparkDataFrame_1979_7 = spark.sql("""
                        SELECT
                        *
                        FROM nasa_ym WHERE (Year == 1979) and (Month == 7)
                        order by lon, lat, Year, Month
                        """)

df_1979_7 = SparkDataFrame_1979_7.toPandas()

scaled_df_1979_7 = MinMaxScaling(df_1979_7, ['Evap', 'Rainf', 'AvgSurfT', 'Albedo', 'SoilM_100_200cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilT_10_40cm'])

output_df_1979_7 = do_kmeans_and_return_df_with_cluster_column(df_1979_7, scaled_df_1979_7, 2, 123)

SparkDataFrame_1979_7_pca = spark.sql("""
                        SELECT
                        *
                        FROM nasa_ym WHERE (Year == 1979) and (Month == 7)
                        order by lon, lat, Year, Month
                        """)

df_1979_7_pca = SparkDataFrame_1979_7_pca.toPandas()

# Standaryzacja
standarized_df_1979_7_pca = StandardScaler().fit_transform(df_1979_7_pca.drop(columns=["Month", "lon", "lat", "Year"]))
# PCA
pca = PCA(n_components=3)
standarized_df_1979_7_pca_done = pca.fit_transform(standarized_df_1979_7_pca)

# Wyjaśniona wariancja
pca.explained_variance_ratio_

standarized_df_1979_7_pca_done = pd.DataFrame(data = standarized_df_1979_7_pca_done, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

output_df_1979_7_pca = do_kmeans_and_return_df_with_cluster_column(df_1979_7_pca, standarized_df_1979_7_pca_done, 2, 123)

SparkDataFrame_1979_1 = spark.sql("""
                        SELECT
                        *
                        FROM nasa_ym WHERE (Year == 1979) and (Month == 1)
                        order by lon, lat, Year, Month
                        """)

df_1979_1 = SparkDataFrame_1979_1.toPandas()

scaled_df_1979_1 = MinMaxScaling(df_1979_1, ['Evap', 'Rainf', 'AvgSurfT', 'Albedo', 'SoilM_100_200cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilT_10_40cm'])

output_df_1979_1 = do_kmeans_and_return_df_with_cluster_column(df_1979_1, scaled_df_1979_1, 2, 123)

SparkDataFrame_1979_1_pca = spark.sql("""
                        SELECT
                        *
                        FROM nasa_ym WHERE (Year == 1979) and (Month == 1)
                        order by lon, lat, Year, Month
                        """)

df_1979_1_pca = SparkDataFrame_1979_1_pca.toPandas()

standarized_df_1979_1_pca = StandardScaler().fit_transform(df_1979_1_pca.drop(columns=["Month", "lon", "lat", "Year"]))

pca = PCA(n_components=2)
standarized_df_1979_1_pca_done = pca.fit_transform(standarized_df_1979_1_pca)

pca.explained_variance_ratio_

standarized_df_1979_1_pca_done = pd.DataFrame(data = standarized_df_1979_1_pca_done, columns = ['principal component 1', 'principal component 2'])

output_df_1979_1_pca = do_kmeans_and_return_df_with_cluster_column(df_1979_1_pca, standarized_df_1979_1_pca_done, 2, 123)

!pip install datashader

!pip install holoviews hvplot colorcet

!pip install geoviews

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

colormap_cluster = get_colormap([0, max(output_df_2023_7.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_2023_7, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 01.07.2023 (wszystkie kolumny)", alpha=0.5))

colormap_cluster = get_colormap([0, max(output_df_2023_7_pca.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_2023_7_pca, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 01.07.2023 (PCA)", alpha=0.5))

colormap_cluster = get_colormap([0, max(output_df_2023_1.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_2023_1, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 01.01.2023 (wszystkie kolumny)", alpha=0.5))

colormap_cluster = get_colormap([0, max(output_df_2023_1_pca.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_2023_1_pca, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 01.01.2023 (PCA)", alpha=0.5))

colormap_cluster = get_colormap([0, max(output_df_2000_7.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_2000_7, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 01.07.2000 (wszystkie kolumny)", alpha=0.5))

colormap_cluster = get_colormap([0, max(output_df_2000_7_pca.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_2000_7_pca, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 01.07.2000 (PCA)", alpha=0.5))

colormap_cluster = get_colormap([0, max(output_df_2000_1.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_2000_1, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 01.01.2000 (wszystkie kolumny)", alpha=0.5))

colormap_cluster = get_colormap([0, max(output_df_2000_1_pca.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_2000_1_pca, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 01.01.2000 (PCA)", alpha=0.5))

colormap_cluster = get_colormap([0, max(output_df_1979_7.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_1979_7, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 01.07.1979 (wszystkie kolumny)", alpha=0.5))

colormap_cluster = get_colormap([0, max(output_df_1979_7_pca.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_1979_7_pca, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 01.07.1979 (PCA)", alpha=0.5))

colormap_cluster = get_colormap([0, max(output_df_1979_1.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_1979_1, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 01.01.1979 (wszystkie kolumny)", alpha=0.5))

colormap_cluster = get_colormap([0, max(output_df_1979_1_pca.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_1979_1_pca, parameter_name='cluster', colormap=colormap_cluster, title="K-średnich dla danych z 01.07.1979 (PCA)", alpha=0.5))

output_df_1979_7.head()

output_df_2000_7.head()

output_df_2023_7.head()

merged_to_2023_7 = copy.deepcopy(output_df_2023_7)
merged_to_2023_7["cluster_2000"] = output_df_2000_7["cluster"].apply(lambda x: 0 if x==1 else 1)
merged_to_2023_7["cluster_1979"] = output_df_1979_7["cluster"]
merged_to_2023_7.head(10)

df_from_not_desert_to_desert = merged_to_2023_7.loc[(merged_to_2023_7["cluster"]==0) & (merged_to_2023_7["cluster_2000"]==1)].dropna()
df_from_not_desert_to_desert

colormap_cluster = get_colormap([0, 1], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=df_from_not_desert_to_desert, parameter_name='cluster', colormap=colormap_cluster, title="Warunek: niepustynia w 01.07.2000 i pustynia w 01.07.2023", alpha=0.5, colorbar_verbose=False))

df_from_not_desert_to_desert = merged_to_2023_7.loc[(merged_to_2023_7["cluster"]==0) & (merged_to_2023_7["cluster_1979"]==1)].dropna()
df_from_not_desert_to_desert

colormap_cluster = get_colormap([0, 1], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=df_from_not_desert_to_desert, parameter_name='cluster', colormap=colormap_cluster, title="Warunek: niepustynia w 01.07.1979 i pustynia w 01.07.2023", alpha=0.5, colorbar_verbose=False))

df_from_not_desert_to_desert = merged_to_2023_7.loc[(merged_to_2023_7["cluster_2000"]==0) & (merged_to_2023_7["cluster_1979"]==1)].dropna()
df_from_not_desert_to_desert

colormap_cluster = get_colormap([0, 1], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=df_from_not_desert_to_desert, parameter_name='cluster', colormap=colormap_cluster, title="Warunek: niepustynia w 01.07.1979 i pustynia w 01.07.2000", alpha=0.5))

NASA_sample_an = pd.read_csv('/content/drive/MyDrive/BigMess/NASA/NASA_an.csv',sep=';')

NASA_sample_an.head()

NASA_sample_an['pustynia_i_step'] = NASA_sample_an.pustynia + NASA_sample_an.step

NASA_sample_an.head(8)

NASA_sample_an["pustynia_i_step"] = NASA_sample_an["pustynia_i_step"].apply(lambda x: 0 if x==1 else 1)

result = merged_to_2023_7.merge(NASA_sample_an, left_on=['lon','lat'], right_on = ['lon','lat'], how='inner')
result

colormap_cluster = get_colormap([0, 1], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=result, parameter_name='pustynia_i_step', colormap=colormap_cluster, title="Zanotowane punkty: 0 - pustynia, 1 - niepustynia", alpha=0.5))

positive = len(result.loc[result.cluster == result.pustynia_i_step])
accuracy = str(round(positive/len(NASA_sample_an)*100,2))
print("Accuracy na poziomie",accuracy+"%.")

positive = len(result.loc[result.cluster != result.pustynia])
accuracy = str(round(positive/len(NASA_sample_an)*100,2))
print("Accuracy na poziomie",accuracy+"%.")

merged_to_2023_7_pca = copy.deepcopy(output_df_2023_7_pca)
merged_to_2023_7_pca["cluster_2000"] = output_df_2000_7_pca["cluster"].apply(lambda x: 0 if x==1 else 1)
merged_to_2023_7_pca["cluster_1979"] = output_df_1979_7_pca["cluster"]
merged_to_2023_7_pca.head(10)

result_pca = merged_to_2023_7_pca.merge(NASA_sample_an, left_on=['lon','lat'], right_on = ['lon','lat'], how='inner')
result_pca

positive = len(result_pca.loc[result_pca.cluster == result_pca.pustynia_i_step])
accuracy = str(round(positive/len(NASA_sample_an)*100,2))
print("Accuracy na poziomie",accuracy+"%.")

positive = len(result_pca.loc[result_pca.cluster != result_pca.pustynia])
accuracy = str(round(positive/len(NASA_sample_an)*100,2))
print("Accuracy na poziomie",accuracy+"%.")
