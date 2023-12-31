!pip install pyspark

from google.colab import drive
from pyspark.sql.functions import col
from pyspark.sql.functions import size
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType, StringType, StructType
from pyspark.sql import Window

import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T

# tworzenie sesji w Sparku
spark = SparkSession.builder.appName('SparkWindows').getOrCreate()

# wczytanie danych z google drive
drive.mount('/content/drive')

columns = ['lon', 'lat', 'Date', 'Rainf', 'Evap', 'AvgSurfT', 'Albedo','SoilT_10_40cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilM_100_200cm']

# Utworzenie schematu okreslajacego typ zmiennych
schema = StructType()
for i in columns:
  if i == "Date":
    schema = schema.add(i, IntegerType(), True)
  else:
    schema = schema.add(i, FloatType(), True)

nasa = spark.read.format('csv').option("header", True).schema(schema).load('/content/drive/MyDrive/BigMess/NASA/NASA.csv')
nasa.show(5)

# rozdzielenie kolumny Date na Year i Month
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

# Funkcja do tworzenia cech czasowych z wykorzystaniem funkcji window w Sparku
def overYearStats(column: str, n: int):
  """
    Funkcja liczaca statystyki srednia i mediane dla wybranej zmiennej z zakresu n miesiecy wstecz i n miesiecy w przod dla danych wspolrzednych geograficznych

    :param column: zmienna, dla ktorej liczymy min, max itd.
    :param n: liczba dni w przod i w tyl z ktorych liczymy statystyki
  """
  windowSpec = Window.partitionBy("lon", "lat").orderBy("Year", "Month").rowsBetween(-n, n)

  nasa_window = (nasa.withColumn("average_" + column, F.avg(F.col(column)).over(windowSpec))
                .withColumn("median_" + column, F.expr("percentile_approx(" + column + ", 0.5)").over(windowSpec))
                )
  return nasa_window

result = overYearStats("GVEG", 5)

result.show(5)

# jako ze korzystamy z funkcji rangeBetween musimy sprawdzic czy dla kazdej pary wspolrzednych jest zapis z 12 miesiecy kazdego roku. Najpierw stworzymy dodatkowe kolumy "allyears" i "allmonths"
# ktore zawierac beda listy wszystkich lat i miesiecy dla danej pary wspolrzednych
windowSpec = Window.partitionBy('lon', 'lat')

test = (result.withColumn('AllYears', F.collect_list(F.col('Year')).over(windowSpec))
       .withColumn('AllMonths', F.collect_list(F.col('Month')).over(windowSpec))
       )

# usuwamy z naszej tabeli pomocniczej duplikaty lat i miesiecy z list w wyzej utworzonych kolumnach
test = (test.withColumn('AllYears', F.array_distinct('AllYears'))
       .withColumn('AllMonths', F.array_distinct('AllMonths'))
       )

test.show()

test_checked_years = test.withColumn("are_years_valid", size(col("AllYears")) == 45) # sprawdzamy czy dla kazdej pary wspolrzednych w kolumnie allYears mamy wszystkie 45 lat
test_checked_months = test.withColumn("are_months_valid", size(col("AllMonths")) == 12) # sprawdzamy czy dla kazdej pary wspolrzednych w kolumnie allMonths mamy wszystkie 12 miesiecy

# Sprawdzenie, czy istnieja jakiekolwiek niewlasciwe tablice w kolumnie "AllYears" i "AllMonths"
invalid_years_count = test_checked_years.filter(col("are_years_valid") == False).count()
invalid_months_count = test_checked_months.filter(col("are_months_valid") == False).count()

# sprwadzamy czy wszedzie mamy 45 lat i 12 miesiecy
print(invalid_years_count, "and", invalid_months_count)
