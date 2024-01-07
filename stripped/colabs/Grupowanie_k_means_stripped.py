"""
# Wczytywanie danych w sparku
"""

"""
Utworzenie środowiska pyspark do obliczeń:
"""

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
import warnings
warnings.filterwarnings('ignore')


"""
Utowrzenie sesji:
"""

spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()


"""
Połączenie z dyskiem:
"""

drive.mount('/content/drive')


"""
Wczytanie danych NASA znajdujących się na dysku w sparku:
"""

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


"""
Zanim zaczniemy pisać kwerendy należy jeszcze dodać nasz DataFrame (df) do "przestrzeni nazw tabel" Sparka:
"""

nasa.createOrReplaceTempView("nasa")


"""
Rozdzielenie kolumny "Date" na kolumny "Year" oraz "Month"
"""

nasa_ym = spark.sql("""
          SELECT
          CAST(SUBSTRING(CAST(Date AS STRING), 1, 4) AS INT) AS Year,
          CAST(SUBSTRING(CAST(Date AS STRING), 5, 2) AS INT) AS Month,
          n.*
          FROM nasa n
          """)
nasa_ym = nasa_ym.drop("Date")


nasa_ym.createOrReplaceTempView("nasa_ym")


"""
# Grupowanie k-średnich
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


def plot_kmeans(scaled_df: pd.DataFrame, n_clusters: int, random_state: int) -> None:
  """
  Funkcja wykonuje grupowanie k-średnich dla 1, 2 ..., n_clusters klastrów oraz tworzy wykres przedstawiający wartości inercji w zależności od zadanej liczby klastrów.
  Parametry:
  - scaled_df (DataFrame): Pandas DataFrame zawierająca przeskalowane kolumny, na podstawie których dokonywane jest grupowanie,
  - n_clusters (int): maksymalna liczba klastrów,
  - random_state (int): ziarno losowości.
  """
  # Wykonanie algorytmu k-średnich dla n_clusters i zapisanie wyników
  SSE = []
  for cluster in range(1, n_clusters):
      kmeans = KMeans(n_clusters = cluster, init='k-means++', random_state=random_state)
      kmeans.fit(scaled_df)
      SSE.append(kmeans.inertia_)

  # Przedstawienie wartości na wykresie
  plt.figure(figsize=(12,6))
  plt.plot(range(1, n_clusters), SSE)
  plt.xlabel('Liczba klastrów')
  plt.ylabel('Suma wariancji')
  plt.title('Wybór optymalnej liczby klastrów - metoda "łokcia"')
  plt.show()


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
## 1. Dane z 01.07.2023
"""

"""
### 1.1. Kolumny: Evap, Rainf, AvgSurfT, Albedo, SoilM_100_200cm, GVEG, PotEvap, RootMoist, SoilT_10_40cm.
"""

"""
Zacznijmy od próby grupowania dla danych z lipca 2023, używając do tego kolumn: Evap, Rainf, AvgSurfT, Albedo, SoilM_100_200cm, GVEG, PotEvap, RootMoist, SoilT_10_40cm.
"""

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


plot_kmeans(scaled_df_2023_7, 8, 123)


output_df_2023_7 = do_kmeans_and_return_df_with_cluster_column(df_2023_7, scaled_df_2023_7, 4, 123)


"""
### 1.2. Kolumny z PCA
"""

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


plot_kmeans(standarized_df_2023_7_pca_done, 8, 123)


output_df_2023_7_pca = do_kmeans_and_return_df_with_cluster_column(df_2023_7_pca, standarized_df_2023_7_pca_done, 4, 123)


"""
## 2. Dane z 01.01.2023
"""

"""
### 2.1 Kolumny: Evap, Rainf, AvgSurfT, Albedo, SoilM_100_200cm, GVEG, PotEvap, RootMoist, SoilT_10_40cm.
"""

SparkDataFrame_2023_1 = spark.sql("""
                        SELECT
                        *
                        FROM nasa_ym WHERE (Year == 2023) and (Month == 1)
                        order by lon, lat, Year, Month
                        """)


df_2023_1 = SparkDataFrame_2023_1.toPandas()


scaled_df_2023_1 = MinMaxScaling(df_2023_1, ['Evap', 'Rainf', 'AvgSurfT', 'Albedo', 'SoilM_100_200cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilT_10_40cm'])


plot_kmeans(scaled_df_2023_1, 8, 123)


output_df_2023_1 = do_kmeans_and_return_df_with_cluster_column(df_2023_1, scaled_df_2023_1, 4, 123)


"""
### 2.2 Kolumny z PCA
"""

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


plot_kmeans(standarized_df_2023_1_pca_done, 8, 123)


output_df_2023_1_pca = do_kmeans_and_return_df_with_cluster_column(df_2023_1_pca, standarized_df_2023_1_pca_done, 4, 123)


"""
## 3. Dane z 01.07.2000
"""

"""
### 3.1. Kolumny: Evap, Rainf, AvgSurfT, Albedo, SoilM_100_200cm, GVEG, PotEvap, RootMoist, SoilT_10_40cm.
"""

SparkDataFrame_2000_7 = spark.sql("""
                        SELECT
                        *
                        FROM nasa_ym WHERE (Year == 2000) and (Month == 7)
                        order by lon, lat, Year, Month
                        """)


df_2000_7 = SparkDataFrame_2000_7.toPandas()


scaled_df_2000_7 = MinMaxScaling(df_2000_7, ['Evap', 'Rainf', 'AvgSurfT', 'Albedo', 'SoilM_100_200cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilT_10_40cm'])


plot_kmeans(scaled_df_2000_7, 8, 123)


output_df_2000_7 = do_kmeans_and_return_df_with_cluster_column(df_2000_7, scaled_df_2000_7, 4, 123)


"""
### 3.2. Kolumny z PCA
"""

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


plot_kmeans(standarized_df_2000_7_pca_done, 8, 123)


output_df_2000_7_pca = do_kmeans_and_return_df_with_cluster_column(df_2000_7_pca, standarized_df_2000_7_pca_done, 4, 123)


"""
## 4. Dane z 01.01.2000
"""

"""
### 4.1. Kolumny: Evap, Rainf, AvgSurfT, Albedo, SoilM_100_200cm, GVEG, PotEvap, RootMoist, SoilT_10_40cm.

"""

SparkDataFrame_2000_1 = spark.sql("""
                        SELECT
                        *
                        FROM nasa_ym WHERE (Year == 2000) and (Month == 1)
                        order by lon, lat, Year, Month
                        """)


df_2000_1 = SparkDataFrame_2000_1.toPandas()


scaled_df_2000_1 = MinMaxScaling(df_2000_1, ['Evap', 'Rainf', 'AvgSurfT', 'Albedo', 'SoilM_100_200cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilT_10_40cm'])


plot_kmeans(scaled_df_2000_1, 8, 123)


output_df_2000_1 = do_kmeans_and_return_df_with_cluster_column(df_2000_1, scaled_df_2000_1, 4, 123)


"""
### 4.2. Kolumny z PCA
"""

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


plot_kmeans(standarized_df_2000_1_pca_done, 8, 123)


output_df_2000_1_pca = do_kmeans_and_return_df_with_cluster_column(df_2000_1_pca, standarized_df_2000_1_pca_done, 4, 123)


"""
## 5. Dane z 01.07.1979
"""

"""
### 5.1. Kolumny: Evap, Rainf, AvgSurfT, Albedo, SoilM_100_200cm, GVEG, PotEvap, RootMoist, SoilT_10_40cm.
"""

SparkDataFrame_1979_7 = spark.sql("""
                        SELECT
                        *
                        FROM nasa_ym WHERE (Year == 1979) and (Month == 7)
                        order by lon, lat, Year, Month
                        """)


df_1979_7 = SparkDataFrame_1979_7.toPandas()


scaled_df_1979_7 = MinMaxScaling(df_1979_7, ['Evap', 'Rainf', 'AvgSurfT', 'Albedo', 'SoilM_100_200cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilT_10_40cm'])


plot_kmeans(scaled_df_1979_7, 8, 123)


output_df_1979_7 = do_kmeans_and_return_df_with_cluster_column(df_1979_7, scaled_df_1979_7, 4, 123)


"""
### 5.2. Kolumny z PCA
"""

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


plot_kmeans(standarized_df_1979_7_pca_done, 8, 123)


output_df_1979_7_pca = do_kmeans_and_return_df_with_cluster_column(df_1979_7_pca, standarized_df_1979_7_pca_done, 4, 123)




"""
## 6. Dane z 01.01.1979
"""

"""
### 6.1. Kolumny: Evap, Rainf, AvgSurfT, Albedo, SoilM_100_200cm, GVEG, PotEvap, RootMoist, SoilT_10_40cm.

"""

SparkDataFrame_1979_1 = spark.sql("""
                        SELECT
                        *
                        FROM nasa_ym WHERE (Year == 1979) and (Month == 1)
                        order by lon, lat, Year, Month
                        """)


df_1979_1 = SparkDataFrame_1979_1.toPandas()


scaled_df_1979_1 = MinMaxScaling(df_1979_1, ['Evap', 'Rainf', 'AvgSurfT', 'Albedo', 'SoilM_100_200cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilT_10_40cm'])


plot_kmeans(scaled_df_1979_1, 8, 123)


output_df_1979_1 = do_kmeans_and_return_df_with_cluster_column(df_1979_1, scaled_df_1979_1, 4, 123)


"""
### 6.2. Kolumny z PCA
"""

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


plot_kmeans(standarized_df_1979_1_pca_done, 8, 123)


output_df_1979_1_pca = do_kmeans_and_return_df_with_cluster_column(df_1979_1_pca, standarized_df_1979_1_pca_done, 4, 123)


"""
# Graficzne przedstawienie klastrów
"""

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


def plot_map(df: pd.DataFrame, parameter_name: str, colormap: mpl.colors.LinearSegmentedColormap,
             point_size: int = 8, width: int = 800, height: int = 500, alpha: float = 1,
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
        colorbar=True,
        toolbar='above',
        tools=['hover', 'wheel_zoom', 'reset'],
        alpha=alpha, # przezroczystość
        bgcolor=bgcolor
    )

    return hv.render(map_with_points)


display(IFrame("https://www.google.com/maps/embed?pb=!1m14!1m12!1m3!1d13982681.959428234!2d-98.66341902257437!3d38.39997874427714!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!5e1!3m2!1spl!2spl!4v1703000232420!5m2!1spl!2spl", '800px', '500px'))


colormap_cluster = get_colormap([0, max(output_df_2023_7.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_2023_7, parameter_name='cluster', colormap=colormap_cluster, alpha=0.5))


colormap_cluster = get_colormap([0, max(output_df_2023_7_pca.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_2023_7_pca, parameter_name='cluster', colormap=colormap_cluster, alpha=0.5))


colormap_cluster = get_colormap([0, max(output_df_2023_1.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_2023_1, parameter_name='cluster', colormap=colormap_cluster, alpha=0.5))


colormap_cluster = get_colormap([0, max(output_df_2023_1_pca.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_2023_1_pca, parameter_name='cluster', colormap=colormap_cluster, alpha=0.5))


colormap_cluster = get_colormap([0, max(output_df_2000_7.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_2000_7, parameter_name='cluster', colormap=colormap_cluster, alpha=0.5))


colormap_cluster = get_colormap([0, max(output_df_2000_7_pca.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_2000_7_pca, parameter_name='cluster', colormap=colormap_cluster, alpha=0.5))


colormap_cluster = get_colormap([0, max(output_df_2000_1.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_2000_1, parameter_name='cluster', colormap=colormap_cluster, alpha=0.5))


colormap_cluster = get_colormap([0, max(output_df_2000_1_pca.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_2000_1_pca, parameter_name='cluster', colormap=colormap_cluster, alpha=0.5))


colormap_cluster = get_colormap([0, max(output_df_1979_7.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_1979_7, parameter_name='cluster', colormap=colormap_cluster, alpha=0.5))


colormap_cluster = get_colormap([0, max(output_df_1979_7_pca.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_1979_7_pca, parameter_name='cluster', colormap=colormap_cluster, alpha=0.5))


colormap_cluster = get_colormap([0, max(output_df_1979_1.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_1979_1, parameter_name='cluster', colormap=colormap_cluster, alpha=0.5))


colormap_cluster = get_colormap([0, max(output_df_1979_1_pca.cluster.values)], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=output_df_1979_1_pca, parameter_name='cluster', colormap=colormap_cluster, alpha=0.5))


output_df_1979_7.head()


output_df_2000_7.head()


output_df_2023_7.head()


"""
# Porównanie punktów z 01.07.2023, 01.07.2000 i 01.07.1979.
"""

merged_to_2023_7 = output_df_2023_7
merged_to_2023_7["cluster_2000"] = output_df_2000_7["cluster"].apply(lambda x: int(0) if x==3 else (int(2) if x==1 else (int(1) if x==2 else int(3))))
merged_to_2023_7["cluster_1979"] = output_df_1979_7["cluster"].apply(lambda x: int(0) if x==3 else (int(2) if x==0 else (int(3) if x==2 else int(1))))
merged_to_2023_7.head(10)


"""
Sprawdźmy, które punkty były zakwalifikowane jako te niepustynne znajdujące się najbliżej pustyni w lipcu 2000, a stały się pustynią w lipcu 2023.
"""

df_from_not_desert_to_desert = merged_to_2023_7.loc[(merged_to_2023_7["cluster"]==0) & (merged_to_2023_7["cluster_2000"]==2)].dropna()


df_from_not_desert_to_desert


colormap_cluster = get_colormap([0, 1], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=df_from_not_desert_to_desert, parameter_name='cluster', colormap=colormap_cluster, alpha=0.5))


"""
Sprawdźmy, które punkty były zakwalifikowane jako te niepustynne znajdujące się najbliżej pustyni w lipcu 1979, a stały się pustynią w lipcu 2023.
"""

df_from_not_desert_to_desert = merged_to_2023_7.loc[(merged_to_2023_7["cluster"]==0) & (merged_to_2023_7["cluster_1979"]==2)].dropna()
df_from_not_desert_to_desert


colormap_cluster = get_colormap([0, 1], ['yellow', 'darkgreen'])
output_notebook()
show(plot_map(df=df_from_not_desert_to_desert, parameter_name='cluster', colormap=colormap_cluster, alpha=0.5))


display(IFrame("https://www.google.com/maps/embed?pb=!1m14!1m12!1m3!1d13982681.959428234!2d-98.66341902257437!3d38.39997874427714!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!5e1!3m2!1spl!2spl!4v1703000232420!5m2!1spl!2spl", '800px', '500px'))




