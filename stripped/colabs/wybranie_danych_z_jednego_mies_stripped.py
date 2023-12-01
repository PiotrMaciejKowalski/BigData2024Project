"""
<a href="https://colab.research.google.com/github/PiotrMaciejKowalski/BigData2024Project/blob/Narzdzie-do-wizualizacji-na-mapach/wybranie_danych_z_jednego_mies.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
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

from pyspark.sql import SparkSession

from google.colab import drive

from pyspark.sql.types import IntegerType, FloatType, StringType, StructType

import pandas as pd


from pyspark.sql import Window
from pyspark.sql.functions import row_number


"""
# Wczytanie zbioru danych
"""

spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()


drive.mount('/content/drive')


# Wczytanie zbioru sampled w celu pobrania nazw kolumn
sampled = pd.read_csv('/content/drive/MyDrive/sampled_NASA_200k.csv')

# Utworzenie schematu określającego typ zmiennych
schemat = StructType()
for i in sampled.columns:
  if i == "Date":
    schemat = schemat.add(i, StringType(), True)
  else:
    schemat = schemat.add(i, FloatType(), True)


# Wczytanie zbioru Nasa w sparku
nasa = spark.read.format('csv').option("header", True).schema(schemat).load('/content/drive/MyDrive/NASA.csv')
nasa.show(5)


nasa.createOrReplaceTempView("nasa")


"""
Na potrzeby przedstawienia danych na mapie zostawimy jedynie dane z grudnia 2022 roku.
"""

nasa_dec22 = spark.sql("""
    SELECT
        CAST(SUBSTRING(CAST(Date AS STRING), 1, 4) AS INT) AS Year,
        CAST(SUBSTRING(CAST(Date AS STRING), 5, 2) AS INT) AS Month,
        n.*
    FROM nasa n
    WHERE CAST(SUBSTRING(CAST(Date AS STRING), 1, 4) AS INT) = 2022
      AND CAST(SUBSTRING(CAST(Date AS STRING), 5, 2) AS INT) = 12
""")


nasa_dec22 = nasa_dec22.drop("Date")
nasa_dec22.show(5)


"""
Zostawmy tylko unikatowe lokalizacje, tj. unikatowe pary wartości lon i lat:
"""

window_spec = Window.partitionBy('lon', 'lat').orderBy('lon', 'lat')
# numerujemy, który raz z kolei dana lokalizacja pojawia się w zbiorze
nasa_dec22_with_loc_row_num = nasa_dec22.withColumn('row_num', row_number().over(window_spec))
nasa_dec22_loc = nasa_dec22_with_loc_row_num.filter('row_num = 1').drop('row_num')


"""
I przenieśmy się na pandas, zapisując od razu wydzielony z oryginalnych danych podzbiór:
"""

nasa_dec22_loc = nasa_dec22_loc.toPandas()
nasa_dec22_loc.to_csv('/content/drive/MyDrive/nasa_dec22_loc.csv', index=False)


