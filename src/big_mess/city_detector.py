#Wymagane: utworzone środowisko pyspark, sesja spark oraz połaczenie z dyskiem
from typing import Optional, Dict
import pandas as pd
from pandas import read_csv as pd_read_csv
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as SparkDataFrame, Column
from pyspark.sql import DataFrame #FIXME should be deleted since duplicated
from pyspark.sql.functions import col, when, isnull, isnan


class City_Geolocalization_Data():

  def __init__(self, path: str) -> None:
    """
    Constructor shall initialize geodata for cities based on given csv file
    internally we shall store it as pandas data frame
    """
    #FIXME add assertions 
    self.geodata = pd.read_csv(path)
    #FIXME confirm that geodata contains no duplicates (the same pair (lon, lat) )

  def get_cityname(
      self,
      lon: float,
      lat: float,
      ) -> Optional[str]:
      #FIXME return None on missing lon, lat pair
      return self.geodata[ (self.geodata.lon == lon) & (self.geodata.lat == lat)].miasto.iloc[0]

  def pandas_bulk_get_geodata(
      self,
      df: pd.DataFrame,
      city: Optional[bool] = True,
      state: Optional[bool] = False,
      country: Optional[bool] = False
      ) -> pd.DataFrame:
    assert "lon" in df.columns
    assert "lat" in df.columns
    _dx = df[['lon', 'lat']]
    columns = []
    if city: columns.append('miasto')
    if state: columns.append('stan')
    if country: columns.append('panstwo')
    if len(columns) == 0:
      return None
    else:
      return _dx.merge(self.geodata[columns],on = ('lon', 'lat'))

  def __pandas_bulk_get_all_geodata( #private version for unbounded data load
    self,
    city: Optional[bool] = True,
    state: Optional[bool] = False,
    country: Optional[bool] = False
    ) -> pd.DataFrame:
    columns = ['lon', 'lat']
    if city: columns.append('miasto')
    if state: columns.append('stan')
    if country: columns.append('panstwo')
    return self.geodata[columns]
  
  def spark_add_geodata(
      self,
      sdf: SparkDataFrame,
      city: Optional[bool] = True,
      state: Optional[bool] = False,
      country: Optional[bool] = False
  ) ->SparkDataFrame:
    
    pandas_geodata = self.__pandas_bulk_get_all_geodata(city=city, state=state, country=country)
    spark_session: SparkSession = sdf.sql_ctx.sparkSession
    spark_geodata = spark_session.createDataFrame(pandas_geodata)
    return sdf.join(spark_geodata, [sdf.lat==spark_geodata.lat, sdf.lon==spark_geodata.lon], 'left')

    

# spark = SparkSession.builder\
#         .master("local")\
#         .appName("Colab")\
#         .config('spark.ui.port', '4050')\
#         .getOrCreate()
        
        
# def add_city(
#     df: DataFrame,
#     path: Optional[str] = "/content/drive/MyDrive/BigMess/NASA/geolokalizacja_wsp_NASA.csv",
#     add_city_flag: Optional[bool] = True,
#     add_city_name: Optional[bool] = False,
#     add_state: Optional[bool] = False,
#     add_country: Optional[bool] = False
# ) -> DataFrame:
#   """
#   The function returns a spark DataFrame in which columns are added according to the flags assigned in the function parameters.
#   One of the function parameters is the "path" variable, it is the path to the .csv file with locations. By default, 'path' is set to a file in the 'BigMess' project directory.
#   Parameters:  
#   add_city_flag - adding a "City_flag" column to the DataFrame with logical values determining whether a given location corresponds to a city.
#   add_city_name - adding a "City" column to the DataFrame with the city name.
#   add_state - adding a "State" column to the DataFrame with the state name.
#   add_country - adding a "Country" column to the DataFrame with the country name.
#   If there is no information in the location file, a NULL value is inserted.
#   """

#   assert "lon" in df.columns
#   assert "lat" in df.columns
#   assert dict(df.dtypes)["lon"] == "float"
#   assert dict(df.dtypes)["lat"] == "float"

#   assert path[len(path)-4:len(path)] == ".csv"

#   #Wczytanie danych
#   location_data = pd_read_csv(path)
#   columns_loc=location_data.columns+'_loc'
#   location_data.columns=columns_loc

#   location_data=spark.createDataFrame(location_data)

#   assert dict(location_data.dtypes)["lon_loc"] in ("float", 'double')
#   assert dict(location_data.dtypes)["lat_loc"] in ("float", 'double')
#   assert "lon_loc" in location_data.columns
#   assert "lat_loc" in location_data.columns
#   assert location_data.count() > 0

#   #Łączenie tabel
#   df_new = df.join(location_data,(df.lon==location_data.lon_loc) & (df.lat==location_data.lat_loc),"left")

#   #DODAWANIE KOLUMN

#   def add_new_column(
#     df: DataFrame,
#     column_loc: str,
#     new_column_name: str,
#     if_true: Column,
#     if_false: Column
#   ) -> DataFrame:
#     assert column_loc in location_data.columns
#     condition = when(~((col(column_loc) =='NA') | isnull(col(column_loc)) | isnan(col(column_loc))),if_true)
#     condition = condition.otherwise(if_false)
#     return df.withColumn(new_column_name,condition)
    

#   #Dodanie flagi czy miasto
#   if add_city_flag:
#     df_new=add_new_column(df_new, 'miasto_loc', 'City_flag', True, False) 

#   #Dodanie miasta
#   if add_city_name:
#     df_new=add_new_column(df_new, 'miasto_loc', 'City', col('miasto_loc'), None) 
  
#   #Dodanie stanu
#   if add_state:
#     df_new=add_new_column(df_new, 'stan_loc', 'State', col('stan_loc'), None) 

#   #Dodanie państwa
#   if add_country:
#     df_new=add_new_column(df_new, 'panstwo_loc', 'Country', col('panstwo_loc'), None) 

#   #Usunięcie kolumn z wczytanej tabeli
#   df_new=df_new.drop(*list(columns_loc))

#   return df_new