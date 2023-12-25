"""
## **Implementacja algorytmu kkN**
"""

"""
### **1. Wczytanie zbioru**
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


from functools import reduce
import math
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np
from geopy.distance import geodesic

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


# Wczytanie zbioru sampled w celu pobrania nazw kolumn
sampled = pd.read_csv('/content/drive/MyDrive/BigMess/NASA/sampled_NASA_200k.csv')

# Utworzenie schematu określającego typ zmiennych
schemat = StructType()
for i in sampled.columns:
  if i == "Date":
    schemat = schemat.add(i, IntegerType(), True)
  else:
    schemat = schemat.add(i, FloatType(), True)


%%time
# Wczytanie zbioru Nasa w sparku
nasa = spark.read.format('csv').option("header", True).schema(schemat).load('/content/drive/MyDrive/BigMess/NASA/NASA.csv')
nasa.show(5)

nasa.createOrReplaceTempView("nasa")

nasa = (
    nasa
    .withColumn('Year', (F.col('Date') / 100).cast('int'))
    .withColumn('Month', F.col('Date') % 100)
    .drop('Date')
)
nasa.show(5)


nasa_coords = spark.sql("""SELECT DISTINCT lat, lon FROM nasa""")
nasa_coords.collect()


"""
### **2. Implementacja algorytmu kNN:**
"""

# function searches for points that lie within a (euclidean) ball of size *radius* around the query points
# if optional argument *k* is given then function searches for (at most) k nearest points that lie within
# a (euclidean) ball of size *radius* around the query points

#point = (latitude, longitude)
def kRadiusNN(df: SparkDataFrame, radius: float, point: Tuple[float, float], k: Optional[int]=None) -> pd.DataFrame :
  assert (25.0625<= point[0] <=52.9375) and (-124.9375<= point[1] <=-67.0625 ), 'Wrong coordinates (out of range)'

  df = df.collect()
  neighbours_pd = pd.DataFrame({"lon":[], "lat":[], 'dist':[]})

  for row in df:

     lon = row['lon']
     lat = row['lat']
     dist = geodesic((lat, lon), point).km
     if dist <= radius:
        new_row = {'lon': lon , 'lat': lat, 'dist' : dist}
        neighbours_pd.loc[len(neighbours_pd)] = new_row

  if k and (k < len(neighbours_pd)):

      neighbours_pd = neighbours_pd.sort_values('dist', ascending=True)
      if not (neighbours_pd.at[k-1,'dist']==neighbours_pd.at[k,'dist']):  #checking if there is no tie (more neighbours with the same distance)
         neighbours_pd = neighbours_pd.iloc[:k]
      else:
         raise Exception("Unable to determine k nearest neighbours: more neighbours with the same distance")

  return(neighbours_pd)


kRadiusNN(nasa_coords, 20, (35, -110), 5)


# weighted: if True then function will weight points by the inverse of their distance (in this case, closer neighbours of
# a query point will have a greater influence than neighbors which are further away).

def predict_class(df: SparkDataFrame, point: Tuple[float, float], radius: float, label_column_name: str,
                  k: Optional[int]=None, weighted: Optional[bool]=False) -> bool :
 if k:
    neighbours = kRadiusNN(df, radius, point, k)
 else:
    neighbours = kRadiusNN(df, radius, point)

 neighbours_rows = df.withColumn('neighbour', F.when( reduce (lambda cond1, cond2: (cond1) | (cond2),
  [(F.col('lat') == neighbours.at[id, 'lat'])&(F.col('lon') == neighbours.at[id, 'lon'])
  for id in neighbours.index]), 1).otherwise(0))

 neighbours_rows = neighbours_rows.filter('neighbour = 1').toPandas()
 neighbours_dist = pd.concat([neighbours_rows, neighbours], axis=1, join="inner")


 if weighted:   #weighted nearest neighbours
    neighbours_dist['dist'] = neighbours_dist['dist'].map(lambda x: 1/x)

    label0 = neighbours_dist[neighbours_dist[label_column_name]==0]
    label1 = neighbours_dist[neighbours_dist[label_column_name]==1]

    frequency0 = label0['dist'].sum()
    frequency1 = label1['dist'].sum()

    return 0 if frequency0 > frequency1 else 1


 else:
    if len(neighbours_dist[label_column_name].mode()) > 1:    #this is only possible when the number of neighbours is an even number (since we perform binary classification)
       raise Exception("Unable to predict the label: we have a tie, there is no clear winner in the majority voting")
    else:
      predicted_label = neighbours_dist[label_column_name].mode().iat[0]

    return predicted_label


"""
Wygenerujemy w sposób sztuczny etykiety dla zbioru w celu przetestowania funkcji:
"""

df = nasa_coords.withColumn('label', (F.col('lat').cast('int'))%2)


df.show(10)


predict_class(df, (40.5, -110), 20, "label", k=None, weighted=True)


predict_class(df, (40.5, -110), 20, "label", k=None, weighted=False)


