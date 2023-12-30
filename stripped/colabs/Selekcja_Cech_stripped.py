!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q dlcdn.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
!tar xf spark-3.5.0-bin-hadoop3.tgz
!pip install -q findspark

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.5.0-bin-hadoop3"

import findspark
findspark.init()
from pyspark.sql.types import IntegerType, FloatType, StringType, StructType
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

import pandas as pd
import matplotlib.pyplot as plt

spark = (
         SparkSession.builder
        .master("local")
        .appName("Colab")
        .config('spark.ui.port', '4050')
        .getOrCreate()
)

from google.colab import drive
drive.mount('/content/drive')

sampled = pd.read_csv('/content/drive/MyDrive/BigMess/NASA/sampled_NASA_200k.csv')

schemat = StructType()
for i in sampled.columns:
  if i == "Date":
    schemat = schemat.add(i, StringType(), True)
  else:
    schemat = schemat.add(i, FloatType(), True)

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

nasa2 = nasa.where((nasa.Year.isin([2009, 2011, 2012, 2013, 2017])))
nasa2.show(5)
#Obszary pustynne:                             #wspolrzedne pustyn i niepustyn pochodza z notatnika dot. klasyfikacji pustyn (nie mam pewnosci, czy sa dobre...)
CD = nasa2.where((nasa2.lon>-104)&(nasa2.lon<-102)&(nasa2.lat>30)&(nasa2.lat< 31)).toPandas()
CP = nasa2.where((nasa2.lon>-110.5)&(nasa2.lon<-108.5)&(nasa2.lat>39)&(nasa2.lat< 40.5)).toPandas()
GBD = nasa2.where((nasa2.lon>-116)&(nasa2.lon<-114)&(nasa2.lat>40)&(nasa2.lat< 41.5)).toPandas()

#Obszary pośrednine:
CD_przejsciowe = nasa2.where((nasa2.lon>-106.5)&(nasa2.lon<-104.5)&(nasa2.lat>32.5)&(nasa2.lat< 33.5)).toPandas()
CP_przejsciowe = nasa2.where((nasa2.lon>-109)&(nasa2.lon<-107)&(nasa2.lat>37.5)&(nasa2.lat< 39)).toPandas()
GBD_przejsciowe = nasa2.where((nasa2.lon>-115)&(nasa2.lon<-113)&(nasa2.lat>42.5)&(nasa2.lat< 44)).toPandas()

#Obszary niepustynne:
niepustynia_obok_CD = nasa2.where((nasa2.lon>-109.5)&(nasa2.lon<-107.5)&(nasa2.lat>33)&(nasa2.lat< 34)).toPandas()
niepustynia_obok_CP = nasa2.where((nasa2.lon>-107)&(nasa2.lon<-105)&(nasa2.lat>9)&(nasa2.lat< 40.5)).toPandas()
niepustynia_obok_GBD = nasa2.where((nasa2.lon>-124)&(nasa2.lon<-122)&(nasa2.lat>40)&(nasa2.lat< 41)).toPandas()

def plot_histogram(ax, data, column, title, color):
    ax.hist(data[column], bins=20, color=color, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.set_xlim(min(min(CD[column]), min(CD_przejsciowe[column]), min(niepustynia_obok_CD[column]),
                    min(CP[column]), min(CP_przejsciowe[column]), min(niepustynia_obok_CP[column]),
                    min(GBD[column]), min(GBD_przejsciowe[column]), min(niepustynia_obok_GBD[column])),
                max(max(CD[column]), max(CD_przejsciowe[column]), max(niepustynia_obok_CD[column]),
                    max(CP[column]), max(CP_przejsciowe[column]), max(niepustynia_obok_CP[column]),
                    max(GBD[column]), max(GBD_przejsciowe[column]), max(niepustynia_obok_GBD[column])))

def plot_feature(column):

  fig, axes = plt.subplots(3, 3, figsize=(15, 15))

  plot_histogram(axes[0, 0], CD, column, "CD_pustynia", "orange")
  plot_histogram(axes[0, 1], CD_przejsciowe, column, "CD_pustynia/nie-pustynia", "yellow")
  plot_histogram(axes[0, 2], niepustynia_obok_CD, column, "CD_nie-pustynia", "green")

  plot_histogram(axes[1, 0], CP, column, "CP_pustynia", "orange")
  plot_histogram(axes[1, 1], CP_przejsciowe, column, "CP_pustynia/nie-pustynia", "yellow")
  plot_histogram(axes[1, 2], niepustynia_obok_CP, column, "CP_nie-pustynia", "green")

  plot_histogram(axes[2, 0], GBD, column, "GBD_pustynia", "orange")
  plot_histogram(axes[2, 1], GBD_przejsciowe, column, "GBD_pustynia/nie-pustynia", "yellow")
  plot_histogram(axes[2, 2], niepustynia_obok_GBD, column, "GBD_nie-pustynia", "green")

  plt.tight_layout()
  plt.show()

plot_feature('SoilM_0_10cm')

print(CD['SoilM_0_10cm'].quantile(0.75), CD_przejsciowe['SoilM_0_10cm'].quantile(0.75), niepustynia_obok_CD['SoilM_0_10cm'].quantile(0.75))
print(CP['SoilM_0_10cm'].quantile(0.75), CP_przejsciowe['SoilM_0_10cm'].quantile(0.75), niepustynia_obok_CP['SoilM_0_10cm'].quantile(0.75))
print(GBD['SoilM_0_10cm'].quantile(0.75), GBD_przejsciowe['SoilM_0_10cm'].quantile(0.75), niepustynia_obok_GBD['SoilM_0_10cm'].quantile(0.75))

plot_feature('SoilM_10_40cm')

plot_feature('SoilM_40_100cm')

plot_feature('SoilM_0_100cm')

plot_feature('SoilM_100_200cm')

plot_feature('RootMoist')

plot_feature('SMLiq_10_40cm')

plot_feature('SMLiq_40_100cm')

plot_feature('SMLiq_100_200cm')

plot_feature('SMAvail_0_200cm')


