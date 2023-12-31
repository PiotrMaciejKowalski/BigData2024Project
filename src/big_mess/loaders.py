from typing import Optional
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame, functions as F
from pyspark.sql.types import IntegerType, FloatType, StringType, StructType


def default_loader(spark: SparkSession, file_path: Optional[str] = None ) -> SparkDataFrame:
    
    columns = ['lon', 'lat', 'Date', 'Rainf', 'Evap', 'AvgSurfT', 'Albedo','SoilT_10_40cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilM_100_200cm']

    if file_path is None:
        file_path = '/content/drive/MyDrive/BigMess/NASA/NASA.csv'
    # Utworzenie schematu określającego typ zmiennych
    schema = StructType()
    for i in columns:
        if i == "Date":
            schema = schema.add(i, IntegerType(), True)
        else:
            schema = schema.add(i, FloatType(), True)

    sdf = spark.read.format('csv').option("header", True).schema(schema).load(file_path)
    return (
        sdf
        .withColumn('Year', sdf['Date'].substr(1, 4).cast('int'))
        .withColumn('Month', sdf['Date'].substr(5, 2).cast('int'))
        .select([F.col(column) for column in sdf.columns if column != 'Date'])
        )
    

def save_to_csv(sdf: SparkDataFrame, output_path: str) -> None:
    """
    Save a PySpark DataFrame to a CSV file.

    Parameters:
    - df: PySpark DataFrame
    - output_path: Output path for the CSV file
    """
    # Save the DataFrame to CSV
    sdf.write.csv(output_path, header=True, mode='overwrite')
