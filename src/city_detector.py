#Wymagane: utworzone środowisko pyspark, sesja spark oraz połaczenie z dyskiem
from pyspark.sql import DataFrame
from pyspark.sql.types import FloatType, StringType
from typing import Optional
from pyspark.sql.functions import col, when, isnull, isnan
from pandas import read_csv as pd_read_csv

def add_city(
    df: DataFrame,
    path: Optional[str] = "/content/drive/MyDrive/BigMess/NASA/geolokalizacja_wsp_NASA.csv",
    add_city_flag: Optional[bool] = True,
    add_city_name: Optional[bool] = False,
    add_state: Optional[bool] = False,
    add_country: Optional[bool] = False
) -> DataFrame:
  """
  Funkcja zwraca sparkowy DataFrame, w którym zostają dodane kolumny według przyznanych flag w parametrach funkcji.
  Jednym z parametrów funkcji jest zmienna "path", jest to ścieżka do pliku .csv z lokalizacjami. Domyślnie ustawiona na plik w wspólnym katalogu porojektu.
  Parametry:  
  add_city_flag - Dodanie kolumny "City_flag" do DataFramu z wartościami logicznymi określającymi, czy dana lokalizacja odpowiada miastu, zgodnie z wczytanym plikiem.
  add_city_name - Dodanie kolumny "City" do DataFramu z nazwą miasta.
  add_state - Dodanie kolumny "State" do DataFramu z nazwą stanu.
  add_country - Dodanie kolumny "Country" do DataFramu z nazwą kraju.
  Jeżeli brak informacji w wczytanym pliku z lokalizacjami, to wstawiana jest wartość NULL.
  """

  assert "lon" in df.columns
  assert "lat" in df.columns
  assert dict(df.dtypes)["lon"] == "float"
  assert dict(df.dtypes)["lat"] == "float"

  assert path[len(path)-4:len(path)] == ".csv"

  #WCZYTWANIE DANYCH
  sampled = pd_read_csv(path)
  # Utworzenie schematu określającego typ zmiennych
  schemat = StructType()
  columns_loc = sampled.columns+'_loc'
  for i in columns_loc:
    if (i == "lon") | (i == "lat"):
      schemat = schemat.add(i, FloatType(), True)
    else:
      schemat = schemat.add(i, StringType(), True)
  #Wczytanie zbioru
  location_data=spark.read.format('csv').option("header", True).schema(schemat).load(path)
  assert "lon_loc" in location_data.columns
  assert "lat_loc" in location_data.columns
  assert location_data.count() > 0

  #Łączenie tabel
  df_new = df.join(location_data,(df.lon==location_data.lon_loc) & (df.lat==location_data.lat_loc),"left")

  #DODAWANIE KOLUMN
  #Dodanie flagi czy miasto
  if add_city_flag == True:
    condition = when(~((col('miasto_loc') =='NA') | isnull(col('miasto_loc')) | isnan(col('miasto_loc'))),True)
    condition = condition.otherwise(False)
    df_new=df_new.withColumn("City_flag",condition)

  #Dodanie miasta
  if add_city_name == True:
    condition = when(~((col('miasto_loc') =='NA') | isnull(col('miasto_loc')) | isnan(col('miasto_loc'))),col('miasto_loc'))
    condition = condition.otherwise(None)
    df_new=df_new.withColumn("City",condition)
  
  #Dodanie stanu
  if add_state == True:
    condition = when(~((col('stan_loc') =='NA') | isnull(col('stan_loc')) | isnan(col('stan_loc'))),col('stan_loc'))
    condition = condition.otherwise(None)
    df_new=df_new.withColumn("State",condition) 

  #Dodanie państwa
  if add_country == True:
    condition = when(~((col('panstwo_loc') =='NA') | isnull(col('panstwo_loc')) | isnan(col('panstwo_loc'))),col('panstwo_loc'))
    condition = condition.otherwise(None)
    df_new=df_new.withColumn("Country",condition)

  #Usunięcie kolumn z wczytanej tabeli
  df_new=df_new.drop(*list(columns_loc))

  return df_new