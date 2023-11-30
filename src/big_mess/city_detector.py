#Wymagane: utworzone środowisko pyspark, sesja spark oraz połaczenie z dyskiem
import pandas as pd
from typing import Optional
from functools import reduce
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as SparkDataFrame

class City_Geolocalization_Data():

  def __init__(self, path: str) -> None:
    """
    Constructor shall initialize geodata for cities based on given csv file
    internally we shall store it as pandas data frame
    """
    assert path[len(path)-4:len(path)] == ".csv"

    self.geodata = pd.read_csv(path)

    assert "lon" in self.geodata.columns
    assert "lat" in self.geodata.columns
    assert dict(self.geodata.dtypes)["lon"] == "float"
    assert dict(self.geodata.dtypes)["lat"] == "float"

    self.geodata.rename(columns = {'miasto':'City', 'stan':'State', 'panstwo':'Country'}, inplace = True)
  
    duplicate_values = self.geodata[['lon','lat']].duplicated()
    assert not reduce(lambda x, y: True if x or y else False, duplicate_values.tolist())

  def get_cityname(
      self,
      lon: float,
      lat: float,
      ) -> Optional[str]:
      if self.geodata[ (self.geodata.lon == lon) & (self.geodata.lat == lat)].empty:
        return None
      else:
        return self.geodata[ (self.geodata.lon == lon) & (self.geodata.lat == lat)].City.iloc[0]

  def __pandas_bulk_get_all_geodata( #private version for unbounded data load
    self,
    city: Optional[bool] = True,
    state: Optional[bool] = False,
    country: Optional[bool] = False
    ) -> pd.DataFrame:
    columns = ['lon', 'lat']
    if city: 
      columns.append('City') 
      assert "City" in self.geodata.columns
    if state: 
      columns.append('State')
      assert "State" in self.geodata.columns
    if country: 
      columns.append('Country')
      assert "Country" in self.geodata.columns
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
    spark_geodata=spark_geodata.withColumnRenamed("lon","lon_geo") \
                               .withColumnRenamed("lat","lat_geo")
    return sdf.join(spark_geodata, [sdf.lat==spark_geodata.lat_geo, sdf.lon==spark_geodata.lon_geo], 'left').drop("lon_geo","lat_geo")