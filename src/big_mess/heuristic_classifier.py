# the file contains the (so called) Rafał "Awesome" Algorithm for desert - no desert distinction

from pyspark.sql import DataFrame as SparkDataFrame, functions as F


def heuristic_classify(data: SparkDataFrame) -> SparkDataFrame:
    less_than_columns = {
        'Rainf': 30,
        'Evap': 33, 
        'GVEG' : 0.333,
    } 
    greater_than_columns = {
        'AvgSurfT': 289,
        'Albedo': 26.7, 
        'SoilT_40_100cm' : 286,
    }
    columns_to_check = list(less_than_columns.keys())+ list(greater_than_columns.keys())
    for column, value in less_than_columns.items():
        data = data.withColumn(f'{column}_condition', F.when(F.col(column) <= value,1).otherwise(0))
    for column, value in greater_than_columns.items():
        data = data.withColumn(f'{column}_condition', F.when(F.col(column) >= value,1).otherwise(0))
    data = data.withColumn("conditions_fullfiled_sum", sum(F.col(column + "_condition") for column in columns_to_check))
    return data.withColumn("Pustynia", F.when(F.col("conditions_fullfiled_sum") >=4, 1).otherwise(0))