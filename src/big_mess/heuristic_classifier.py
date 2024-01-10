# the file contains the (so called) Rafał "Awesome" Algorithm for desert - no desert distinction

from pyspark.sql import DataFrame as SparkDataFrame, functions as F


def heuristic_classify(data: SparkDataFrame) -> SparkDataFrame:
    assert all([_ for column in data.columns if '_condition' not in column ]), "the data must not contain condition in any column name"
    less_than_columns = {
        "Rainf": 30,
        "Evap": 33,
        "GVEG": 0.333,
    }
    greater_than_columns = {
        "AvgSurfT": 289,
        "Albedo": 26.7,
        "SoilT_40_100cm": 286,
    }
    minimal_condition_count = 4
    columns_to_check = list(less_than_columns.keys()) + list(
        greater_than_columns.keys()
    )
    for column, value in less_than_columns.items():
        data = data.withColumn(
            f"{column}_condition", F.when(F.col(column) <= value, 1).otherwise(0)
        )
    for column, value in greater_than_columns.items():
        data = data.withColumn(
            f"{column}_condition", F.when(F.col(column) >= value, 1).otherwise(0)
        )
    return (
        data.withColumn(
            "conditions_fullfiled_sum",
            sum(F.col(column + "_condition") for column in columns_to_check),
        )
        .withColumn(
            "Pustynia",
            F.when(
                F.col("conditions_fullfiled_sum") >= minimal_condition_count, 1
            ).otherwise(0),
        )
        .drop("conditions_fullfiled_sum", *[column for column in data.columns if '_condition' in column])
    )
