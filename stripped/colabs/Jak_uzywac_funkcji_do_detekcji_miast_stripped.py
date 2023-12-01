"""
<a href="https://colab.research.google.com/github/PiotrMaciejKowalski/BigData2024Project/blob/Konstrukcja_detektora_miast/Jak_uzywac_funkcji_do_detekcji_miast.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
Dokument ma na celu wytłumaczenie na przykładzie w jaki sposób korzystać z klasy **City_Geolocalization_Data** oraz jak skonfigurować GitHuba w Colabie.
"""

"""
# Opis klasy
"""

"""
**KONSTRUKTOR** \
**City_Geolocalization_Data(path)** - *path*, to ścieżka do pliku .csv z lokalizacjami.

**FUNKCJE** \
**City_Geolocalization_Data.get_cityname(lon, lat)** \
Zwraca nazwę miasta dla podanych współrzędnych lon i lat. W przypadku braku współrzędnych zwróci None.

**City_Geolocalization_Data.spark_add_geodata(sdf, city = True, state = False, country = False)** \
Zwraca Spark DataFrame, w którym zostają dodane kolumny według przyznanych flag w parametrach funkcji. \
Parametry:  \
*sdf* - Spark DataFrame do którego dodajemy kolumny. \
*city* - Flaga dodania kolumny *City* do DataFramu z nazwą miasta. \
*state* - Flaga dodania kolumny *State* do DataFramu z nazwą stanu. \
*country* - Flaga dodania kolumny *Country* do DataFramu z nazwą kraju.
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
!git checkout Konstrukcja_detektora_miast
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

columns = ['lon', 'lat', 'Date', 'SWdown', 'LWdown', 'SWnet', 'LWnet', 'Qle', 'Qh', 'Qg', 'Qf', 'Snowf', 'Rainf', 'Evap', 'Qs', 'Qsb', 'Qsm', 'AvgSurfT', 'Albedo', 'SWE', 'SnowDepth', 'SnowFrac', 'SoilT_0_10cm', 'SoilT_10_40cm',
           'SoilT_40_100cm', 'SoilT_100_200cm', 'SoilM_0_10cm', 'SoilM_10_40cm', 'SoilM_40_100cm', 'SoilM_100_200cm', 'SoilM_0_100cm', 'SoilM_0_200cm', 'RootMoist', 'SMLiq_0_10cm', 'SMLiq_10_40cm', 'SMLiq_40_100cm', 'SMLiq_100_200cm',
           'SMAvail_0_100cm', 'SMAvail_0_200cm', 'PotEvap', 'ECanop', 'TVeg', 'ESoil', 'SubSnow', 'CanopInt', 'ACond', 'CCond', 'RCS', 'RCT', 'RCQ', 'RCSOL', 'RSmin','RSMacr', 'LAI', 'GVEG', 'Streamflow']

# Utworzenie schematu określającego typ zmiennych
schema = StructType()
for i in columns:
  if i == "Date":
    schema = schema.add(i, StringType(), True)
  else:
    schema = schema.add(i, FloatType(), True)


nasa = spark.read.format('csv').option("header", True).schema(schema).load('/content/drive/MyDrive/BigMess/NASA/NASA.csv')
nasa.show(5)


"""
## 2. Użycie funkcji
"""

"""
Stworzenie obiektu będącego realizacją klasy City_Geolocalization_Data.
"""

geodata_file = "/content/drive/MyDrive/BigMess/NASA/geolokalizacja_wsp_NASA.csv"

geo = City_Geolocalization_Data(geodata_file)


"""
Użycie funkcji *get_cityname*.
"""

geo.get_cityname(-78.0625, 25.0625)


"""
Użycie funkcji *spark_add_geodata*.
"""

result = geo.spark_add_geodata(nasa, city=True, state=True, country=True)
result.show(5)


result.select('lon','lat','City','State','Country').filter(result.City != 'NaN').limit(10).toPandas()




