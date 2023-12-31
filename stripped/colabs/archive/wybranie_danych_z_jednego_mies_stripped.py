!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q dlcdn.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
!tar xf spark-3.5.0-bin-hadoop3.tgz
!pip install -q findspark

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.5.0-bin-hadoop3"

import findspark
findspark.init()

from pyspark.sql import SparkSession

from google.colab import drive

from pyspark.sql.types import IntegerType, FloatType, StringType, StructType

import pandas as pd

spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()

drive.mount('/content/drive')

# Wczytanie zbioru sampled w celu pobrania nazw kolumn
sampled = pd.read_csv('/content/drive/MyDrive/BigMess/NASA/sampled_NASA_200k.csv')

# Utworzenie schematu określającego typ zmiennych
schemat = StructType()
for i in sampled.columns:
  if i == "Date":
    schemat = schemat.add(i, StringType(), True)
  else:
    schemat = schemat.add(i, FloatType(), True)

# Wczytanie zbioru Nasa w sparku
nasa = spark.read.format('csv').option("header", True).schema(schemat).load('/content/drive/MyDrive/BigMess/NASA/NASA.csv')
nasa.show(5)

nasa.createOrReplaceTempView("nasa")

nasa = spark.sql("""
    SELECT
        CAST(SUBSTRING(CAST(Date AS STRING), 1, 4) AS INT) AS Year,
        CAST(SUBSTRING(CAST(Date AS STRING), 5, 2) AS INT) AS Month,
        n.*
    FROM nasa n
    """)

nasa = nasa.drop("Date")
nasa.show(5)

nasa_dec22_loc = nasa.where((nasa.Year==2022) & (nasa.Month==12))

nasa_dec22_loc = nasa_dec22_loc.toPandas()

nasa_dec22_loc = nasa_dec22_loc.groupby(['lon', 'lat']).size().reset_index(name='count')
max_count = nasa_dec22_loc['count'].max()
max_count

nasa_dec22_loc = nasa_dec22_loc.drop('count', axis=1)

nasa_dec22_loc.to_csv('/content/drive/MyDrive/BigMess/NASA/nasa_dec22_loc.csv', index=False)


