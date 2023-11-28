"""
<a href="https://colab.research.google.com/github/PiotrMaciejKowalski/BigData2024Project/blob/detektor_miast_korekta/colabs/Jak_uzywac_funkcji_do_detekcji_miast.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

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
# Setup repozytorium i środowiska
"""

"""
Najpierw instalacja sparka
"""

!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q dlcdn.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
!tar xf spark-3.5.0-bin-hadoop3.tgz
!pip install -q findspark


"""
Dalej importujemy nasz projekt do zasobów. Będzie go widać po lewej stronie w zakładce files
"""

!git clone https://github.com/PiotrMaciejKowalski/BigData2024Project.git


"""
*Odkomentować i ustawić odpowiedni branch dla danego kodu*
"""

%cd BigData2024Project/
!git checkout detektor_miast_korekta
%cd ..


"""
Pakiet poetry jest potrzebny aby zbudować paczkę z naszego projektu big_mess. To jej budowy potrzebny jest plik `pyproject.toml`
"""

!pip install poetry


"""
Gdy mamy zainstalowane poetry użyjemy go do zbudowania pakietu
"""

%cd BigData2024Project/
!poetry install
%cd ..


"""
i jego lokalnego zainstalowania
"""

%cd BigData2024Project/
!pip install -e .
%cd ..


"""
Potem występuje trudny krok teraz (prawdopodobnie) trzeba **ZRESTARTOWAĆ RUNTIME**. Wybieramy sekcje Runtime i Restart runtime
"""

"""
Okey - dalej konfigurejemy sparka
"""

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.5.0-bin-hadoop3"

import findspark
findspark.init()


"""
Wszystko gotowe - można importować
"""

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType, StringType, StructType

from google.colab import drive

from big_mess.city_detector import City_Geolocalization_Data


"""
# Przykład użycia
"""

"""
## 1. Wczytanie danych w sparku
"""

"""
Utworzenie środowiska pyspark do obliczeń:
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
"""

geodata_file = "/content/drive/MyDrive/BigMess/NASA/geolokalizacja_wsp_NASA.csv"

geo= City_Geolocalization_Data(geodata_file)


geo.get_cityname(-78.0625, 25.0625)


"""
With spark
"""

result = geo.spark_add_geodata(nasa.limit(5), city=True, state=True, country=True)
result.show()


