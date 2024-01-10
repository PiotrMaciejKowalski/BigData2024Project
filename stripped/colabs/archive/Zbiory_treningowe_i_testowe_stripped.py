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
import random
import math
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np

from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql.types import IntegerType, FloatType, StringType, StructType, StructField, Row
from pyspark.sql import functions as F

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
nasa.show()

#max_year = nasa.agg(max('Year').alias("max_year")).first()["max_year"]
max_year = 2023

prediction_test_set = nasa.where(nasa.Year >= max_year-1)

%%time
prediction_test_set.show(5)

%%time
prediction_train_set = nasa.where(nasa.Year < max_year-1)
prediction_train_set.show(5)

size_test1 = prediction_test_set.count()
size_train1 = prediction_train_set.count()

print(f'We have {size_train1} records in train - which is {size_train1/(size_train1+size_test1)*100:3.1f}%')
print(f'We have {size_test1} records in train - which is {size_test1/(size_train1+size_test1)*100:3.1f}%')

prediction_test_set.write.parquet('/content/drive/MyDrive/BigMess/prediction_test_set.parquet')
prediction_train_set.write.parquet('/content/drive/MyDrive/BigMess/prediction_train_set.parquet')

def assert_rectangles(rectangles: List[Tuple[Tuple[float, float], Tuple[float,float]]]) -> None:
  for lower_left, upper_right in rectangles:
    assert lower_left[0] < upper_right[0] and lower_left[1] < upper_right[1], f'Wrong Georange LL: {lower_left} vs UR: {upper_right}'

def split_with_rectangles(df: SparkDataFrame, rectangles: List[Tuple[Tuple[float, float], Tuple[float,float]]]) -> SparkDataFrame:
  assert_rectangles(rectangles=rectangles)
  return (
      df.withColumn('TEST_ITEM',
                  F.when( reduce( lambda cond1, cond2: (cond1) | (cond2), [
                      ((lower_left[0] < F.col('lon'))
                      & (F.col('lon') < upper_right[0])
                      & (lower_left[1] < F.col('lat'))
                      & (F.col('lat') < upper_right[1]))
                      for lower_left, upper_right
                      in rectangles
                  ])
                  , 1).otherwise(0)
                )
  )


min_lat = 25.0625
max_lat = 52.9375
min_lon = -124.9375
max_lon = -67.0625

area = (max_lat-40)*(max_lon-min_lon)*0.93 + 0.20*(30-min_lat)*(max_lon-min_lon) + 0.5*(40-30)*(max_lon-(-80)) +0.93*(40-30)*(-80-min_lon)

print(0.15*area)

size_train1*0.10  # approx. minimum size of a test set

rectangles = [
     ((-122, 25.5), (-115, 37)),
     ((-124, 37), (-122, 52)),
     ((-93, 24.5), (-79.5, 34)),
     ((-79.5, 34), (-75, 37)),
     ((-115, 25.5), (-111, 30))
     ]

%%time
splitted_dataset = split_with_rectangles(prediction_train_set, rectangles)

%%time
splitted_dataset.show(5)

classification_train_set = splitted_dataset.filter('TEST_ITEM = 0')
classification_train_set.show(5)

classification_test_set = splitted_dataset.filter('TEST_ITEM = 1')
classification_test_set.show(5)

classification_test_set.drop('TEST_ITEM').show(5)
classification_train_set.drop('TEST_ITEM').show(5)


%%time
size_test2 = classification_test_set.count()
size_train2 = classification_train_set.count()
print(f'We have {size_train2} records in train - which is {size_train2/(size_train2+size_test2)*100:3.0f}%')
print(f'We have {size_test2} records in test - which is {size_test2/(size_train2+size_test2)*100:3.0f}%')

classification_test_set.write.parquet('/content/drive/MyDrive/BigMess/classification_test_set.parquet')      #nie odpaliłam zapisu do pliku, bo po zapisaniu wcześniejszych zbiorów przekroczono limit miejsca na Dysku Google...
classification_train_set.write.parquet('/content/drive/MyDrive/BigMess/classification_train_set.parquet')

rectangle = [((-120, 25.06),(-79, 33))]

splitted_dataset2 = split_with_rectangles(prediction_train_set, rectangle)

classification_test_set2 =  splitted_dataset2.filter('TEST_ITEM = 1')
classification_test_set2.show(5)

classification_test_set2.drop('TEST_ITEM').show(5)

classification_train_set2 = splitted_dataset2.filter('TEST_ITEM = 0')
classification_train_set2.show(5)
classification_train_set2.drop('TEST_ITEM').show(5)

size_test22 = classification_test_set2.count()
size_train22 = classification_train_set2.count()

print(f'We have {size_train22} records in train - which is {size_train22/(size_train22+size_test22)*100:3.2f}%')
print(f'We have {size_test22} records in test - which is {size_test22/(size_train22+size_test22)*100:3.2f}%')

rectangle = [((-106, 33),(-80.5, 40))]

splitted_dataset3 = split_with_rectangles(prediction_train_set, rectangle)

classification_test_set3 =  splitted_dataset3.filter('TEST_ITEM = 1')
classification_test_set3.show(5)

classification_train_set3 = splitted_dataset3.filter('TEST_ITEM = 0')
classification_train_set3.show(5)

classification_test_set3.drop('TEST_ITEM').show(5)
classification_train_set3.drop('TEST_ITEM').show(5)

size_test23 = classification_test_set3.count()
size_train23 = classification_train_set3.count()

print(f'We have {size_train23} records in train - which is {size_train23/(size_train23+size_test23)*100:3.2f}%')
print(f'We have {size_test23} records in test - which is {size_test23/(size_train23+size_test23)*100:3.2f}%')

#grid_cell_size - the grid cell size (the length of the side of a square) in degrees
def get_grid(grid_cell_size: float) -> List[Tuple[Tuple[float, float], Tuple[float,float]]] :

 min_lat = 25.0625
 max_lat = 52.9375
 min_lon = -124.9375
 max_lon = -67.0625

 area = (max_lat-min_lat)*(max_lon-min_lon)
 cells_num = area//(grid_cell_size*grid_cell_size)
 actual_grid_size = math.sqrt(area/cells_num)

 xx=np.arange(min_lon, max_lon, step = actual_grid_size)
 yy=np.arange(min_lat, max_lat, step = actual_grid_size)
 cells = []

 for j in range(len(yy)-1):
     for i in range(len(xx)-1):
        cells.append(tuple([tuple([float(xx[i]), float(yy[j])]), tuple([float(xx[i+1]), float(yy[j+1])])]))

 return cells



splitted = split_with_rectangles(prediction_train_set, [((-115, 25.5), (-111, 31))])
splitted.show()

nasa2 = splitted.filter('TEST_ITEM = 0')
block_with_desert = splitted.filter('TEST_ITEM = 1')

nasa2.show(5)
block_with_desert.show(5)

nasa2.drop('TEST_ITEM').show(5)
block_with_desert.drop('TEST_ITEM').show(5)



#k - number of folds
def get_folds(k : int, blocks_cells: List[Tuple[Tuple[float, float], Tuple[float,float]]]) -> List[List[Tuple[Tuple[float, float], Tuple[float,float]]]] :

    n = len(blocks_cells)//k
    reminder = len(blocks_cells)%k
    folds=[]

    for i in range(k):
          fold = []
          for j in range(n):
                r = random.randint(0, len(blocks_cells)-1)
                fold.append(blocks_cells[r])
                blocks_cells.remove(blocks_cells[r])
          folds.append(fold)

    if reminder!=0:
       for b in range(len(blocks_cells)):
           n_fol = random.randint(0, k-1)
           folds[n_fol].append(blocks_cells[b])

    return(folds)

cells = get_grid(2)
folds = get_folds(6, cells)
num_str = [str(i) for i in range(6)]

for i in range(len(folds)):
      fdf = split_with_rectangles(nasa2, folds[i])
      fdf.show()
      fdf.filter('TEST_ITEM = 1').show()
      fdf.drop('TEST_ITEM').show()
      fdf.write.parquet('/content/drive/MyDrive/BigMess/folds/fold'+ num_str[i] + '.parquet')

