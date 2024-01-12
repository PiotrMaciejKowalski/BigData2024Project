from typing import Optional
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql.types import IntegerType, FloatType, StructType


def default_loader(
    spark: SparkSession, file_path: Optional[str] = None
) -> SparkDataFrame:

    columns = [
        "lon",
        "lat",
        "Date",
        "Rainf",
        "Evap",
        "AvgSurfT",
        "Albedo",
        "SoilT_40_100cm",
        "GVEG",
        "PotEvap",
        "RootMoist",
        "SoilM_100_200cm",  
    ]

    if file_path is None:
        file_path = "/content/drive/MyDrive/BigMess/NASA/NASA.csv"
    # Utworzenie schematu określającego typ zmiennych
    schema = StructType()
    for i in columns:
        if i == "Date":
            schema = schema.add(i, IntegerType(), True)
        else:
            schema = schema.add(i, FloatType(), True)

    sdf = spark.read.format("csv").option("header", True).schema(schema).load(file_path)
    return (
        sdf.withColumn("Year", sdf["Date"].substr(1, 4).cast("int"))
        .withColumn("Month", sdf["Date"].substr(5, 2).cast("int"))
        .drop("Date")
    )


def load_single_month(
    spark: SparkSession,
    file_path: Optional[str] = None,
    year: int = 2023,
    month: int = 1,
) -> SparkDataFrame:
    data = default_loader(spark, file_path)
    return data.filter(f"Year = {year}").filter(f"Month = {month}")


def load_anotated(
    spark: SparkSession,
    file_path: Optional[str] = None,
    anotatation_path: Optional[str] = None,
) -> SparkDataFrame:
    data = load_single_month(spark, file_path)
    if anotatation_path is None:
        anotatation_path = "/content/drive/MyDrive/BigMess/NASA/NASA_an.csv"
    annotations = (
        spark.read.format("csv")
        .option("header", True)
        .option("inferSchema", "true")
        .option("delimiter", ";")
        .load(anotatation_path)
    )
    return data.join(annotations, on=["lon", "lat"], how="inner")


def save_to_csv(sdf: SparkDataFrame, output_path: str) -> None:
    """
    Save a PySpark DataFrame to a CSV file.

    Parameters:
    - df: PySpark DataFrame
    - output_path: Output path for the CSV file
    """
    # Save the DataFrame to CSV
    (
        sdf.coalesce(1)  # coalesce(1) is used to reduce the number of partitions
        # to 1, effectively combining the data into a single partition before
        # saving. This ensures that the output will be a single CSV file.
        .write.option("delimiter", ";").csv(output_path, header=True, mode="overwrite")
    )


def preprocessed_loader(
    spark: SparkSession, file_path: Optional[str] = None
) -> SparkDataFrame:
    return (
        spark.read.format("csv")
        .option("header", True)
        .option("inferSchema", "true")
        .option("delimiter", ";")
        .load(file_path)
    )
