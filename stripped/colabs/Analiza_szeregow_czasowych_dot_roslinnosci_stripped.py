"""
<a href="https://colab.research.google.com/github/PiotrMaciejKowalski/BigData2024Project/blob/Analiza-szeregow-czasowych-dot-roslinnosci/colabs/Analiza_szereg%C3%B3w_czasowych_dot_ro%C5%9Blinno%C5%9Bci.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

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
from pyspark.sql.types import IntegerType, FloatType, StringType, StructType


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
# Analiza szergów czasowych  dot. roślinności
"""

# Ograniczenie zbioru nasa do wartości, które potrzebujemy, ustawienie pomiaru zawsze na 1 dzień miesiąca
time_series = spark.sql("""
                        SELECT
                        lon, lat,
                        to_date(CONCAT(Year, '-', Month, '-1')) as Date, GVEG
                        FROM nasa_ym
                        order by lon, lat, Year, Month
                        """)


# Wyzanczenie unikatowych par współrzednych ze zbioru Nasa i zapisanie w Pandas
%%time
distinct_wsp = spark.sql("""
                          SELECT DISTINCT lon, lat FROM nasa
                          """).toPandas()


distinct_wsp.shape


# FIXME przenieść do katalogu src / @see (https://app.clickup.com/t/86bwj0nt4)
def time_series_GVEG(lon: float, lat: float):
  """
  Funkcja generująca wykres w czasie wartości zmiennej GVEG (green vegetation) dla zadanych współrzednych.

  Parametry:
  - lon (float): długość geograficzna
  - lat (float): szerokość geograficzna
  """
  # ograniczenie zbioru do konkretnej pary współrzędnych
  series = time_series.filter((time_series['lon'] == lon) & (time_series['lat'] == lat)).drop('lon', 'lat')
  # Przejście na pandas
  time_series_Pandas = series.toPandas()
  # Ustawienie 'date' jako indeksu
  time_series_Pandas.set_index('Date', inplace=True)
  # Obliczanie 12-miesięcznej średniej kroczącej
  time_series_Pandas['12m_MA'] = time_series_Pandas['GVEG'].rolling(window=12).mean()
  # Tworzenie wykresu
  plt.figure(figsize=(11,6))
  gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
  xmin, xmax = time_series_Pandas.index.min() - pd.DateOffset(years=1) , time_series_Pandas.index.max() + pd.DateOffset(years=1)
  ax1 = plt.subplot(gs[0])
  ax1.plot(time_series_Pandas.index, time_series_Pandas['GVEG']) 
  ax1.plot(time_series_Pandas.index, time_series_Pandas['12m_MA'], label='12-miesięczna średnia krocząca', color='red')
  ax1.set_title(f'Wykres Szeregu Czasowego (lon {lon}, lat {lat})')
  ax1.legend(loc='upper left')
  ax1.set_ylabel('Wartość GVEG')
  ax1.set_xlim(xmin, xmax)
  ax1.grid(True)

  ax2 = plt.subplot(gs[1])
  ax2.plot(time_series_Pandas.index, time_series_Pandas['12m_MA'], label='12-miesięczna średnia krocząca', color='red')
  ax2.set_xlabel('Data')
  ax2.grid(True)
  ax2.set_xlim(xmin, xmax)

  plt.tight_layout()
  plt.show()


# losujemy 10 par wspołrzednych ze zbioru nasa
random = distinct_wsp.sample(10)


# sprawdzenie szeregu czasowego dla wylosowanych współrzędnych
for index, row in random.iterrows():
  time_series_GVEG(row['lon'], row['lat'])


# pustynia mojave i jej "obrzeża" (ograniczenie terenu do współrzednych wytyczających skrajne krańce pustyni)
mojave = distinct_wsp[(distinct_wsp['lon'] >= -116.15) &  (distinct_wsp['lon']  <= -114.95) & (distinct_wsp['lat']  <= 35.45) & (distinct_wsp['lat']  >= 34.7)]


# losujemy 10 współrzednych z zakresu pustyni mojave i jej obrzeży
random_mojave = mojave.sample(10)


# sprawdzenie szeregu czasowego dla wylosowanych współrzędnych
for index, row in random_mojave.iterrows():
  time_series_GVEG(row['lon'], row['lat'])




