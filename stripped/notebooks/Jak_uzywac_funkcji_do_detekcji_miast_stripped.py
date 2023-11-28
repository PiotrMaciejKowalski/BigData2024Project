"""
Dokument ma na celu wytłumaczenie na przykładzie w jaki sposób korzystać z funkcji ***add_city*** służącej do detekcji miast.
"""

"""
# Opis funkcji
"""

"""
Funkcja ***add_city*** zwraca sparkowy DataFrame, w którym zostają dodane kolumny według przyznanych flag w parametrach funkcji.
Jednym z parametrów funkcji jest *path*, jest to ścieżka do pliku .csv z lokalizacjami. Domyślnie ustawiona na plik w wspólnym katalogu projektu. \
Parametry:  \
***add_city_flag*** - Dodanie kolumny *City_flag* do DataFramu z wartościami logicznymi określającymi, czy dana lokalizacja odpowiada miastu, zgodnie z wczytanym plikiem. \
***add_city_name*** - Dodanie kolumny *City* do DataFramu z nazwą miasta. \
***add_state*** - Dodanie kolumny *State* do DataFramu z nazwą stanu. \
***add_country*** - Dodanie kolumny *Country* do DataFramu z nazwą kraju. \
Jeżeli brak informacji w wczytanym pliku z lokalizacjami, to wstawiana jest wartość NULL.
"""

"""
# Przykład użycia
"""

"""
Przed użyciem funkcji konieczne jest utworzenie środowiska w sparku oraz połączenie z dyskiem google.
"""

"""
## 1. Wczytanie danych w sparku
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


import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType, StringType, StructType

from google.colab import drive


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

# Wczytanie zbioru sampled w celu pobrania nazw kolumn
sampled = pd.read_csv('/content/drive/MyDrive/BigMess/NASA/sampled_NASA_200k.csv')

# Utworzenie schematu określającego typ zmiennych
schemat = StructType()
for i in sampled.columns:
  if i == "Date":
    schemat = schemat.add(i, StringType(), True)
  else:
    schemat = schemat.add(i, FloatType(), True)


nasa = spark.read.format('csv').option("header", True).schema(schemat).load('/content/drive/MyDrive/BigMess/NASA/NASA.csv')
nasa.show(5)


"""
## 2. Użycie funkcji
"""

"""
W tym przykładzie funkcja zostanie zaimportowana z katalogu na dysku. Jednak po dodaniu pliku do głównej gałęzi na GitHubie będzie można ją pobierać
stamtąd bezpośrednio.
"""

import sys
sys.path.insert(0,"/content/drive/MyDrive/Colab Notebooks/Analiza BIG DATA/Sprint 1")


import city_detector


"""
Dodamy wszystkie możliwe kolumny.
"""

nasa=city_detector.add_city(nasa, add_city_name=True,add_state=True, add_country=True)


nasa


"""
W obiekcie nasa widzimy dodanie nowych kolumn.
"""

nasa.show(4)


"""
Przykładowe wyniki.
"""

nasa.select('lon','lat','City_flag','City','City_flag','State','Country').limit(10).toPandas()


nasa.select('lon','lat','City_flag','City','City_flag','State','Country').filter(nasa.City_flag==True).limit(10).toPandas()




