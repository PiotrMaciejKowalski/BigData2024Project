!pip install datashader
!pip install holoviews hvplot colorcet
!pip install geoviews

# import doinstalowanych pakietów
import datashader as ds
#import datashader.transfer_functions as tf
import colorcet as cc
import holoviews as hv
#from holoviews.operation.datashader import datashade
import geoviews as gv
import geoviews.tile_sources as gts
from holoviews import opts
from bokeh.plotting import show, output_notebook
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from google.colab import drive
drive.mount("/content/drive")

!git clone https://github.com/PiotrMaciejKowalski/BigData2024Project.git
%cd BigData2024Project
#!git checkout refactoring-sprint2
%cd ..

%cd BigData2024Project
!git pull
%cd ..

%cd BigData2024Project
!git status
%cd ..

!chmod 755 /content/BigData2024Project/src/setup.sh
!/content/BigData2024Project/src/setup.sh

import sys
sys.path.append('/content/BigData2024Project/src')

from start_spark import initialize_spark
initialize_spark()

from pyspark.sql import SparkSession
from big_mess.loaders import preprocessed_loader, load_single_month, save_to_csv
from big_mess.heuristic_classifier import heuristic_classify

def plot_map(df: pd.DataFrame, parameter_name: str, colormap: mpl.colors.LinearSegmentedColormap,
             title: str, point_size: int = 8, width: int = 900, height: int = 600, alpha: float = 1,
             bgcolor: str = 'white'):

    gdf = gv.Points(df, ['lon', 'lat'], [parameter_name]) # obiekt zawierający punkty
    tiles = gts.OSM # wybór mapy tła, w tym wypadku OpenStreetMap

    # łączenie mapy tła z punktami i ustawienie wybranych parametrów wizualizacji
    map_with_points = tiles * gdf.opts(
        title=title,
        color=parameter_name,
        cmap=colormap,
        size=point_size,
        width=width,
        height=height,
        colorbar=True,
        toolbar='above',
        tools=['hover', 'wheel_zoom', 'reset'],
        alpha=alpha, # przezroczystość
        bgcolor=bgcolor
    )

    map_with_points.opts(bgcolor=bgcolor)

    # zapis mapy do pliku .html
    output_filename = f'/content/drive/MyDrive/BigMess/NASA/output_map_{parameter_name}.html'
    hv.save(map_with_points, output_filename)

    return hv.render(map_with_points)

def show_metrics(y_true, y_pred):
  print("Macierz błędu\n", confusion_matrix(y_true, y_pred), "\n")
  accuracy = accuracy_score(y_true, y_pred)
  print(f"Dokładność (accuracy): {accuracy*100:.2f}%")
  precision = precision_score(y_true, y_pred)
  print(f"Precyzja (precision): {precision*100:.2f}%")
  recall = recall_score(y_true, y_pred)
  print(f"Czułość (recall): {recall*100:.2f}%")
  f1 = f1_score(y_true, y_pred)
  print(f"F1-score: {f1*100:.2f}%")

spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()

NASA_sample_an = pd.read_csv('/content/drive/MyDrive/BigMess/NASA/NASA_an.csv', sep=';')

NASA_sample_an.head(10)

NASA_sample_an_points = NASA_sample_an.copy()
NASA_sample_an_points['klasyfikacja'] = 'niepustynia'
NASA_sample_an_points.loc[NASA_sample_an_points['pustynia'] == 1, 'klasyfikacja'] = 'pustynia'
NASA_sample_an_points.loc[NASA_sample_an_points['step'] == 1, 'klasyfikacja'] = 'step'
output_notebook()
show(plot_map(df=NASA_sample_an_points, parameter_name='klasyfikacja',
              colormap=dict(zip(['pustynia', 'niepustynia', 'step'], ['yellow', 'green', 'orange'])),
              title="Zaanotowane punkty", point_size=6, alpha=0.7))

NASA_sample_an['pustynia_i_step'] = NASA_sample_an.pustynia + NASA_sample_an.step

NASA_sample_an.head(10)

# NASA_jan2023 = load_single_month(spark, year=2023, month=1)
# NASA_jan2023_heuristic = heuristic_classify(NASA_jan2023)
# save_to_csv(NASA_jan2023_heuristic, '/content/drive/MyDrive/BigMess/NASA/months_preprocessed/heuristic/NASA_jan2023_heuristic.csv')
NASA_jan2023_heuristic = preprocessed_loader(spark,'/content/drive/MyDrive/BigMess/NASA/months_preprocessed/heuristic/NASA_jan2023_heuristic.csv')

NASA_jan2023_heuristic = NASA_jan2023_heuristic.select("lon", "lat", "Pustynia").toPandas()
NASA_jan2023_heuristic

output_notebook()
show(plot_map(df=NASA_jan2023_heuristic, parameter_name='Pustynia',
              colormap=dict(zip(['1', '0'], ['yellow', 'green'])),
              title='Pustynie (1) i niepustynie (0) wg Algorytmu Nieomylnego (styczeń 2023)',
              point_size=3, alpha=0.7))

NASA_jan2023_an_merge = NASA_jan2023_heuristic.merge(NASA_sample_an, left_on=['lon','lat'], right_on=['lon','lat'], how='inner')
NASA_jan2023_an_merge

show_metrics(NASA_jan2023_an_merge.pustynia_i_step, NASA_jan2023_an_merge.Pustynia)

show_metrics(NASA_jan2023_an_merge.pustynia, NASA_jan2023_an_merge.Pustynia)

NASA_jan2023_an_without_steps = NASA_jan2023_an_merge[NASA_jan2023_an_merge['step'] != 1]
show_metrics(NASA_jan2023_an_without_steps.pustynia_i_step, NASA_jan2023_an_without_steps.Pustynia)

# NASA_april2023 = load_single_month(spark, year=2023, month=4)
# NASA_april2023_heuristic = heuristic_classify(NASA_april2023)
# save_to_csv(NASA_april2023_heuristic, '/content/drive/MyDrive/BigMess/NASA/months_preprocessed/heuristic/NASA_april2023_heuristic.csv')
NASA_april2023_heuristic = preprocessed_loader(spark,'/content/drive/MyDrive/BigMess/NASA/months_preprocessed/heuristic/NASA_april2023_heuristic.csv')

NASA_april2023_heuristic = NASA_april2023_heuristic.select("lon", "lat", "Pustynia").toPandas()
NASA_april2023_heuristic

output_notebook()
show(plot_map(df=NASA_april2023_heuristic, parameter_name='Pustynia',
              colormap=dict(zip(['1', '0'], ['yellow', 'green'])),
              title='Pustynie (1) i niepustynie (0) wg Algorytmu Nieomylnego (kwiecień 2023)',
              point_size=3, alpha=0.7))

NASA_april2023_an_merge = NASA_april2023_heuristic.merge(NASA_sample_an, left_on=['lon','lat'], right_on=['lon','lat'], how='inner')
NASA_april2023_an_merge

show_metrics(NASA_april2023_an_merge.pustynia_i_step, NASA_april2023_an_merge.Pustynia)

show_metrics(NASA_april2023_an_merge.pustynia, NASA_april2023_an_merge.Pustynia)

NASA_sample_an_without_steps = NASA_april2023_an_merge[NASA_april2023_an_merge['step'] != 1]
show_metrics(NASA_sample_an_without_steps.pustynia, NASA_sample_an_without_steps.Pustynia)

# NASA_august2023 = load_single_month(spark, year=2023, month=8)
# NASA_august2023_heuristic = heuristic_classify(NASA_august2023)
# save_to_csv(NASA_august2023_heuristic, '/content/drive/MyDrive/BigMess/NASA/months_preprocessed/heuristic/NASA_august2023_heuristic.csv')
NASA_august2023_heuristic = preprocessed_loader(spark,'/content/drive/MyDrive/BigMess/NASA/months_preprocessed/heuristic/NASA_august2023_heuristic.csv')

NASA_august2023_heuristic = NASA_august2023_heuristic.select("lon", "lat", "Pustynia").toPandas()
NASA_august2023_heuristic

output_notebook()
show(plot_map(df=NASA_august2023_heuristic, parameter_name='Pustynia',
              colormap=dict(zip(['1', '0'], ['yellow', 'green'])),
              title='Pustynie (1) i niepustynie (0) wg Algorytmu Nieomylnego (sierpień 2023)',
              point_size=3, alpha=0.7))

NASA_august2023_an_merge = NASA_august2023_heuristic.merge(NASA_sample_an, left_on=['lon','lat'], right_on=['lon','lat'], how='inner')
NASA_august2023_an_merge

show_metrics(NASA_august2023_an_merge.pustynia_i_step, NASA_august2023_an_merge.Pustynia)

show_metrics(NASA_august2023_an_merge.pustynia, NASA_august2023_an_merge.Pustynia)

NASA_sample_an_without_steps = NASA_august2023_an_merge[NASA_august2023_an_merge['step'] != 1]
show_metrics(NASA_sample_an_without_steps.pustynia, NASA_sample_an_without_steps.Pustynia)

# NASA_oct2022 = load_single_month(spark, year=2022, month=10)
# NASA_oct2022_heuristic = heuristic_classify(NASA_oct2022)
# save_to_csv(NASA_oct2022_heuristic, '/content/drive/MyDrive/BigMess/NASA/months_preprocessed/heuristic/NASA_oct2022_heuristic.csv')
NASA_oct2022_heuristic = preprocessed_loader(spark,'/content/drive/MyDrive/BigMess/NASA/months_preprocessed/heuristic/NASA_oct2022_heuristic.csv')

NASA_oct2022_heuristic = NASA_oct2022_heuristic.select("lon", "lat", "Pustynia").toPandas()
NASA_oct2022_heuristic

output_notebook()
show(plot_map(df=NASA_oct2022_heuristic, parameter_name='Pustynia',
              colormap=dict(zip(['1', '0'], ['yellow', 'green'])),
              title='Pustynie (1) i niepustynie (0) wg Algorytmu Nieomylnego (październik 2022)',
              point_size=3, alpha=0.7))

NASA_oct2022_an_merge = NASA_oct2022_heuristic.merge(NASA_sample_an, left_on=['lon','lat'], right_on=['lon','lat'], how='inner')
NASA_oct2022_an_merge

show_metrics(NASA_oct2022_an_merge.pustynia_i_step, NASA_oct2022_an_merge.Pustynia)

show_metrics(NASA_oct2022_an_merge.pustynia, NASA_oct2022_an_merge.Pustynia)

NASA_sample_an_without_steps = NASA_oct2022_an_merge[NASA_oct2022_an_merge['step'] != 1]
show_metrics(NASA_sample_an_without_steps.pustynia, NASA_sample_an_without_steps.Pustynia)

NASA_sample = pd.read_csv('/content/drive/MyDrive/sampled_NASA_200k.csv')

NASA_sample['Date'] = NASA_sample['Date'].astype(str)
NASA_zima = NASA_sample[~NASA_sample['Date'].str[-2:].isin(["05", "06", "07", "08", "09", "10"])]

selected_columns = ["lon", "lat", "Rainf", "Evap", "AvgSurfT", "Albedo", "SoilT_40_100cm", "GVEG"]
NASA_zima = NASA_zima[selected_columns].dropna()

NASA_zima.head(5)

CD = NASA_zima[(NASA_zima['lon'] >= -104) & (NASA_zima['lon'] <= -102) & (NASA_zima['lat'] >= 30) & (NASA_zima['lat'] <= 31)]
CP = NASA_zima[(NASA_zima['lon'] >= -110.5) & (NASA_zima['lon'] <= -108.5) & (NASA_zima['lat'] >= 39) & (NASA_zima['lat'] <= 40.5)]
GBD = NASA_zima[(NASA_zima['lon'] >= -116) & (NASA_zima['lon'] <= -114) & (NASA_zima['lat'] >= 40) & (NASA_zima['lat'] <= 41.5)]

CD_i_niepustynia = NASA_zima[(NASA_zima['lon'] >= -106.5) & (NASA_zima['lon'] <= -104.5) & (NASA_zima['lat'] >= 32.5) & (NASA_zima['lat'] <= 33.5)]
CP_i_niepustynia = NASA_zima[(NASA_zima['lon'] >= -109) & (NASA_zima['lon'] <= -107) & (NASA_zima['lat'] >= 37.5) & (NASA_zima['lat'] <= 39)]
GBD_i_niepustynia = NASA_zima[(NASA_zima['lon'] >= -115) & (NASA_zima['lon'] <= -113) & (NASA_zima['lat'] >= 42.5) & (NASA_zima['lat'] <= 44)]

niepustynia_przy_CD = NASA_zima[(NASA_zima['lon'] >= -109.5) & (NASA_zima['lon'] <= -107.5) & (NASA_zima['lat'] >= 33) & (NASA_zima['lat'] <= 34)]
niepustynia_przy_CP = NASA_zima[(NASA_zima['lon'] >= -107) & (NASA_zima['lon'] <= -105) & (NASA_zima['lat'] >= 39) & (NASA_zima['lat'] <= 40.5)]
niepustynia_przy_GBD = NASA_zima[(NASA_zima['lon'] >= -124) & (NASA_zima['lon'] <= -122) & (NASA_zima['lat'] >= 39.5) & (NASA_zima['lat'] <= 41)]

Rainf_graniczne = round(
    pd.Series([
        CD_i_niepustynia['Rainf'].quantile(0.75),
        CD_i_niepustynia['Rainf'].quantile(0.25),
        CP_i_niepustynia['Rainf'].quantile(0.75),
        CP_i_niepustynia['Rainf'].quantile(0.25),
        GBD_i_niepustynia['Rainf'].quantile(0.75),
        GBD_i_niepustynia['Rainf'].quantile(0.25),
    ]).mean()
)

Rainf_graniczne

GVEG_graniczne = round(
    pd.Series([
        CD['GVEG'].quantile(0.75),
        niepustynia_przy_CD['GVEG'].quantile(0.25),
        CP['GVEG'].quantile(0.75),
        niepustynia_przy_CP['GVEG'].quantile(0.25),
        GBD['GVEG'].quantile(0.75),
        niepustynia_przy_GBD['GVEG'].quantile(0.25),
    ]).mean(), 3
)

GVEG_graniczne

Evap_graniczne = round(
    pd.Series([
        CD['Evap'].quantile(0.75),
        niepustynia_przy_CD['Evap'].quantile(0.25),
        CP['Evap'].quantile(0.75),
        niepustynia_przy_CP['Evap'].quantile(0.25),
        GBD['Evap'].quantile(0.75),
        niepustynia_przy_GBD['Evap'].quantile(0.25),
    ]).mean()
)

Evap_graniczne

AvgSurfT_graniczne = round(
    pd.Series([
        CD['AvgSurfT'].quantile(0.25),
        niepustynia_przy_CD['AvgSurfT'].quantile(0.75),
        CP['AvgSurfT'].quantile(0.25),
        niepustynia_przy_CP['AvgSurfT'].quantile(0.75),
        GBD['AvgSurfT'].quantile(0.25),
        niepustynia_przy_GBD['AvgSurfT'].quantile(0.75),
    ]).mean()
)

AvgSurfT_graniczne

Albedo_graniczne = round(
    pd.Series([
        CD['Albedo'].quantile(0.25),
        niepustynia_przy_CD['Albedo'].quantile(0.75),
        CP['Albedo'].quantile(0.25),
        niepustynia_przy_CP['Albedo'].quantile(0.75),
        GBD['Albedo'].quantile(0.25),
        niepustynia_przy_GBD['Albedo'].quantile(0.75),
    ]).mean(), 1
)

Albedo_graniczne

SoilT_40_100cm_graniczne = round(
    pd.Series([
        CD['SoilT_40_100cm'].quantile(0.25),
        niepustynia_przy_CD['SoilT_40_100cm'].quantile(0.75),
        CP['SoilT_40_100cm'].quantile(0.25),
        niepustynia_przy_CP['SoilT_40_100cm'].quantile(0.75),
        GBD['SoilT_40_100cm'].quantile(0.25),
        niepustynia_przy_GBD['SoilT_40_100cm'].quantile(0.75),
    ]).mean()
)

SoilT_40_100cm_graniczne

from pyspark.sql import DataFrame as SparkDataFrame, functions as F

def heuristic_classify_winter(data: SparkDataFrame) -> SparkDataFrame:
    assert all(
        [column for column in data.columns if "_condition" not in column]
    ), "the data must not contain condition in any column name"
    less_than_columns = {
        "Rainf": 12,
        "Evap": 13,
        "GVEG": 0.211,
    }
    greater_than_columns = {
        "AvgSurfT": 273,
        "Albedo": 33.4,
        "SoilT_40_100cm": 275,
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
        .drop(
            "conditions_fullfiled_sum",
            *[column for column in data.columns if "_condition" in column],
        )
    )

# NASA_jan2023 = load_single_month(spark, year=2023, month=1)
# NASA_jan2023_heuristic_winter = heuristic_classify_winter(NASA_jan2023)
save_to_csv(NASA_jan2023_heuristic_winter, '/content/drive/MyDrive/BigMess/NASA/months_preprocessed/heuristic/NASA_jan2023_heuristic_winter.csv')
NASA_jan2023_heuristic_winter = preprocessed_loader(spark,'/content/drive/MyDrive/BigMess/NASA/months_preprocessed/heuristic/NASA_jan2023_heuristic_winter.csv')

NASA_jan2023_heuristic_winter = NASA_jan2023_heuristic_winter.select("lon", "lat", "Pustynia").toPandas()
NASA_jan2023_heuristic_winter

output_notebook()
show(plot_map(df=NASA_jan2023_heuristic_winter, parameter_name='Pustynia',
              colormap=dict(zip(['1', '0'], ['yellow', 'green'])),
              title='Pustynie (1) i niepustynie (0) wg Zimowego Algorytmu Nieomylnego (styczeń 2023)',
              point_size=3, alpha=0.7))

NASA_jan2023_an_merge_winter = NASA_jan2023_heuristic_winter.merge(NASA_sample_an, left_on=['lon','lat'], right_on=['lon','lat'], how='inner')
NASA_jan2023_an_merge_winter

show_metrics(NASA_jan2023_an_merge_winter.pustynia_i_step, NASA_jan2023_an_merge_winter.Pustynia)

show_metrics(NASA_jan2023_an_merge_winter.pustynia, NASA_jan2023_an_merge_winter.Pustynia)

NASA_jan2023_an_without_steps_winter = NASA_jan2023_an_merge_winter[NASA_jan2023_an_merge_winter['step'] != 1]
show_metrics(NASA_jan2023_an_without_steps_winter.pustynia_i_step, NASA_jan2023_an_without_steps_winter.Pustynia)

# NASA_april2023 = load_single_month(spark, year=2023, month=4)
# NASA_april2023_heuristic_winter = heuristic_classify_winter(NASA_april2023)
# save_to_csv(NASA_april2023_heuristic_winter, '/content/drive/MyDrive/BigMess/NASA/months_preprocessed/heuristic/NASA_april2023_heuristic_winter.csv')
NASA_april2023_heuristic_winter = preprocessed_loader(spark,'/content/drive/MyDrive/BigMess/NASA/months_preprocessed/heuristic/NASA_april2023_heuristic_winter.csv')

NASA_april2023_heuristic_winter = NASA_april2023_heuristic_winter.select("lon", "lat", "Pustynia").toPandas()
NASA_april2023_heuristic_winter

output_notebook()
show(plot_map(df=NASA_april2023_heuristic_winter, parameter_name='Pustynia',
              colormap=dict(zip(['1', '0'], ['yellow', 'green'])),
              title='Pustynie (1) i niepustynie (0) wg Zimowego Algorytmu Nieomylnego (kwiecień 2023)',
              point_size=3, alpha=0.7))

NASA_april2023_an_merge_winter = NASA_april2023_heuristic_winter.merge(NASA_sample_an, left_on=['lon','lat'], right_on=['lon','lat'], how='inner')
NASA_april2023_an_merge_winter

show_metrics(NASA_april2023_an_merge_winter.pustynia_i_step, NASA_april2023_an_merge_winter.Pustynia)

show_metrics(NASA_april2023_an_merge_winter.pustynia, NASA_april2023_an_merge_winter.Pustynia)

NASA_sample_an_without_steps_winter = NASA_april2023_an_merge_winter[NASA_april2023_an_merge_winter['step'] != 1]
show_metrics(NASA_sample_an_without_steps_winter.pustynia, NASA_sample_an_without_steps_winter.Pustynia)

# NASA_august2023 = load_single_month(spark, year=2023, month=8)
# NASA_august2023_heuristic_winter = heuristic_classify_winter(NASA_august2023)
# save_to_csv(NASA_august2023_heuristic_winter, '/content/drive/MyDrive/BigMess/NASA/months_preprocessed/heuristic/NASA_august2023_heuristic_winter.csv')
NASA_august2023_heuristic_winter = preprocessed_loader(spark,'/content/drive/MyDrive/BigMess/NASA/months_preprocessed/heuristic/NASA_august2023_heuristic_winter.csv')

NASA_august2023_heuristic_winter = NASA_august2023_heuristic_winter.select("lon", "lat", "Pustynia").toPandas()
NASA_august2023_heuristic_winter

output_notebook()
show(plot_map(df=NASA_august2023_heuristic_winter, parameter_name='Pustynia',
              colormap=dict(zip(['1', '0'], ['yellow', 'green'])),
              title='Pustynie (1) i niepustynie (0) wg Zimowego Algorytmu Nieomylnego (sierpień 2023)',
              point_size=3, alpha=0.7))

NASA_august2023_an_merge_winter = NASA_august2023_heuristic_winter.merge(NASA_sample_an, left_on=['lon','lat'], right_on=['lon','lat'], how='inner')
NASA_august2023_an_merge_winter

show_metrics(NASA_august2023_an_merge_winter.pustynia_i_step, NASA_august2023_an_merge_winter.Pustynia)

show_metrics(NASA_august2023_an_merge_winter.pustynia, NASA_august2023_an_merge_winter.Pustynia)

NASA_sample_an_without_steps_winter = NASA_august2023_an_merge_winter[NASA_august2023_an_merge_winter['step'] != 1]
show_metrics(NASA_sample_an_without_steps_winter.pustynia, NASA_sample_an_without_steps_winter.Pustynia)

# NASA_oct2022 = load_single_month(spark, year=2022, month=10)
# NASA_oct2022_heuristic_winter = heuristic_classify_winter(NASA_oct2022)
# save_to_csv(NASA_oct2022_heuristic_winter, '/content/drive/MyDrive/BigMess/NASA/months_preprocessed/heuristic/NASA_oct2022_heuristic_winter.csv')
NASA_oct2022_heuristic_winter = preprocessed_loader(spark,'/content/drive/MyDrive/BigMess/NASA/months_preprocessed/heuristic/NASA_oct2022_heuristic_winter.csv')

NASA_oct2022_heuristic_winter = NASA_oct2022_heuristic_winter.select("lon", "lat", "Pustynia").toPandas()
NASA_oct2022_heuristic_winter

output_notebook()
show(plot_map(df=NASA_oct2022_heuristic_winter, parameter_name='Pustynia',
              colormap=dict(zip(['1', '0'], ['yellow', 'green'])),
              title='Pustynie (1) i niepustynie (0) wg Zimowego Algorytmu Nieomylnego (październik 2022)',
              point_size=3, alpha=0.7))

NASA_oct2022_an_merge_winter = NASA_oct2022_heuristic_winter.merge(NASA_sample_an, left_on=['lon','lat'], right_on=['lon','lat'], how='inner')
NASA_oct2022_an_merge_winter

show_metrics(NASA_oct2022_an_merge_winter.pustynia_i_step, NASA_oct2022_an_merge_winter.Pustynia)

show_metrics(NASA_oct2022_an_merge_winter.pustynia, NASA_oct2022_an_merge_winter.Pustynia)

NASA_sample_an_without_steps_winter = NASA_oct2022_an_merge_winter[NASA_oct2022_an_merge_winter['step'] != 1]
show_metrics(NASA_sample_an_without_steps_winter.pustynia, NASA_sample_an_without_steps_winter.Pustynia)
