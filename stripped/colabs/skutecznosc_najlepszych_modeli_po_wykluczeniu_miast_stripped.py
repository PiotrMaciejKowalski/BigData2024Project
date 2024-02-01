!pip install datashader
!pip install holoviews hvplot colorcet
!pip install geoviews

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib as mpl
import datashader as ds
import datashader.transfer_functions as tf
import colorcet as cc
import holoviews as hv
from holoviews.operation.datashader import datashade
import geoviews as gv
import geoviews.tile_sources as gts
from holoviews import opts
from IPython.display import IFrame
from IPython.core.display import display
from bokeh.plotting import show, output_notebook
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import pickle
import seaborn as sns
import sklearn
from sklearn import tree
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, classification_report, jaccard_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import mutual_info_classif
from typing import Optional, Tuple, List
from pandas import DataFrame
from imblearn.over_sampling import RandomOverSampler, SMOTE
from bokeh.layouts import row
import joblib

from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/PiotrMaciejKowalski/BigData2024Project.git

!chmod 755 /content/BigData2024Project/src/setup.sh
!/content/BigData2024Project/src/setup.sh

import sys
sys.path.append('/content/BigData2024Project/src')

from start_spark import initialize_spark
initialize_spark()

from big_mess.loaders import default_loader
from big_mess.city_detector import City_Geolocalization_Data
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame, functions as F

spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()

nasa = default_loader(spark)

geo_city = City_Geolocalization_Data("/content/drive/MyDrive/BigMess/NASA/geolokalizacja_wsp_NASA.csv")
nasa_city = geo_city.spark_add_geodata(nasa, city=True, state=False, country=False)
nasa_city.createOrReplaceTempView("nasa_city")

geo_anotated = spark.read.format('csv').option("header", True).option("inferSchema", "true").option("delimiter", ";").load('/content/drive/MyDrive/BigMess/NASA/NASA_an.csv')
geo_anotated.createOrReplaceTempView("geo_anotated")

class BalanceDataSet():
  '''
  Two techniques for handling imbalanced data.
  '''
  def __init__(
      self,
      X: DataFrame,
      y: DataFrame
      ) -> None:
      self.X = X
      self.y = y
      assert len(self.X)==len(self.y)

  def useOverSampling(
      self,
      randon_seed: Optional[int] = 2023
      ) -> Tuple[DataFrame, DataFrame]:
    oversample = RandomOverSampler( sampling_strategy='auto',
                  random_state=randon_seed)
    return oversample.fit_resample(self.X, self.y)

  def useSMOTE(
      self,
      randon_seed: Optional[int] = 2023
      ) -> Tuple[DataFrame, DataFrame]:
    smote = SMOTE(random_state=randon_seed)
    return smote.fit_resample(self.X, self.y)

def summary_model(model: sklearn.base.BaseEstimator, X: pd.DataFrame, y: pd.DataFrame, labels_names: List) -> None:
  y_pred = model.predict(X)
  y_real= y
  cf_matrix = confusion_matrix(y_real, y_pred)
  group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
  group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
  labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
  labels = np.asarray(labels).reshape(len(labels_names),len(labels_names))
  sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Reds',xticklabels=labels_names,yticklabels=labels_names)
  plt.xlabel('Predykcja')
  plt.ylabel('Rzeczywistość')
  plt.show()

%cd BigData2024Project
!git checkout agregacje_kilkuletnie
sys.path.append('/content/BigData2024Project/src/big_mess')
from agg_classification_eval import  plot_map, show_metrics, prepare_agg_data_for_training

display(IFrame("https://www.google.com/maps/embed?pb=!1m14!1m12!1m3!1d13982681.959428234!2d-98.66341902257437!3d38.39997874427714!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!5e1!3m2!1spl!2spl!4v1703000232420!5m2!1spl!2spl", '800px', '500px'))



nasa_city_with_anotated_07_2023 = spark.sql("""
          SELECT
          nc.lon, nc.lat, Rainf, Evap, AvgSurfT, Albedo, SoilT_40_100cm, GVEG, PotEvap, RootMoist, SoilM_100_200cm,
          CASE WHEN City = 'NaN' THEN 0 ELSE 1 END AS city,
          pustynia, step
          FROM nasa_city nc
          LEFT JOIN geo_anotated ga ON ga.lon = nc.lon AND ga.lat = nc.lat
          WHERE Year = 2023 AND Month = 7
          """)

df_07_2023 = nasa_city_with_anotated_07_2023.toPandas()

df_07_2023_anotated_without_city = df_07_2023[(df_07_2023['pustynia'].notna()) & (df_07_2023['city'] == 0)]
X = df_07_2023_anotated_without_city.loc[:,'Rainf':'SoilM_100_200cm']
y = df_07_2023_anotated_without_city['pustynia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=321)
X_train_bal, y_train_bal = BalanceDataSet(X_train, y_train).useSMOTE()

X_lat = df_07_2023_anotated_without_city.loc[:,'lat':'SoilM_100_200cm']
X_train_lat, X_test_lat, y_train_lat, y_test_lat = train_test_split(X_lat, y, test_size=0.2, random_state=321)
X_train_lat_bal, y_train_lat_bal = BalanceDataSet(X_train_lat, y_train_lat).useSMOTE()

model_las_m2_7_path = '/content/drive/MyDrive/BigMess/NASA/Modele/Klasyfikacja/Lasy/las_m2_7'
model_las_m2_7 = joblib.load(model_las_m2_7_path)

df_07_2023_predict = df_07_2023.copy()
df_07_2023_predict['Pustynia_predict'] = model_las_m2_7.predict(df_07_2023_predict.loc[:,'Rainf':'SoilM_100_200cm'])
df_07_2023_predict = df_07_2023_predict.sort_values(by=['lon', 'lat'])

params_gs_las = {'n_estimators': [10,20,30,40,50,60,70,80,90,100,110],
                  'max_depth': np.arange(7,11),
                  'min_samples_leaf': np.arange(4,8)}

gs_las = GridSearchCV(RandomForestClassifier(random_state = 2024), cv = 5, param_grid = params_gs_las, scoring = 'accuracy')
gs_las.fit(X_train_bal, y_train_bal)

las = RandomForestClassifier(random_state = 2024, **gs_las.best_params_)
las.fit(X_train_bal, y_train_bal)

summary_model(las, X_train_bal, y_train_bal, ['0','1'])

show_metrics(las, X_train_bal, y_train_bal)

summary_model(las, X_test, y_test, ['0','1'])

show_metrics(las, X_test, y_test)

df_07_2023_predict_without_city = df_07_2023.copy()
df_07_2023_predict_without_city['Pustynia_predict'] = las.predict(df_07_2023_predict_without_city.loc[:,'Rainf':'SoilM_100_200cm'])
df_07_2023_predict_without_city.loc[df_07_2023_predict_without_city['city'] == 1, 'Pustynia_predict'] = 0
df_07_2023_predict_without_city= df_07_2023_predict_without_city.sort_values(by=['lon', 'lat'])

colormap_cluster = dict(zip(['1', '0'], ['darkorange', 'lightyellow']))
plot_with_city  = plot_map(df=df_07_2023_predict, parameter_name='Pustynia_predict', colormap=colormap_cluster, title="Las losowy (lipiec 2023) - model z miastami", point_size=3, alpha=0.5)
plot_without_city = plot_map(df=df_07_2023_predict_without_city, parameter_name='Pustynia_predict', colormap=colormap_cluster, title="Las losowy (lipiec 2023) - model bez miast", point_size=3, alpha=0.5)
output_notebook()
show(row(plot_with_city, plot_without_city))

model_lgbm_m2_7_path ='/content/drive/MyDrive/BigMess/NASA/Modele/Klasyfikacja/LightGBM/lgbm_m2_7'
model_lgbm_m2_7 = joblib.load(model_lgbm_m2_7_path)

df_07_2023_predict_lgbm = df_07_2023.copy()
df_07_2023_predict_lgbm['Pustynia_predict'] = model_lgbm_m2_7.predict(df_07_2023_predict_lgbm.loc[:,'Rainf':'SoilM_100_200cm'])
df_07_2023_predict_lgbm = df_07_2023_predict_lgbm.sort_values(by=['lon', 'lat'])

params_gs_lgbm = {'n_estimators':  [10,50,100,120],
                  'num_leaves': [10,20,30,40],
                  'max_depth': np.arange(5,11)}

gs_lgbm= GridSearchCV(LGBMClassifier(random_state = 2024), cv = 5, param_grid = params_gs_lgbm, scoring = 'accuracy')
gs_lgbm.fit(X_train_bal, y_train_bal)

lgbm = LGBMClassifier(random_state = 2024, **gs_lgbm.best_params_)
lgbm.fit(X_train_bal, y_train_bal)

summary_model(lgbm, X_train_bal, y_train_bal, ['0','1'])

show_metrics(lgbm, X_train_bal, y_train_bal)

summary_model(lgbm, X_test, y_test, ['0','1'])

show_metrics(lgbm, X_test, y_test)

df_07_2023_predict_without_city_lgbm = df_07_2023.copy()
df_07_2023_predict_without_city_lgbm['Pustynia_predict'] = lgbm.predict(df_07_2023_predict_without_city_lgbm.loc[:,'Rainf':'SoilM_100_200cm'])
df_07_2023_predict_without_city_lgbm.loc[df_07_2023_predict_without_city_lgbm['city'] == 1, 'Pustynia_predict'] = 0
df_07_2023_predict_without_city_lgbm= df_07_2023_predict_without_city_lgbm.sort_values(by=['lon', 'lat'])

plot_with_city_lgbm  = plot_map(df=df_07_2023_predict_lgbm, parameter_name='Pustynia_predict', colormap=colormap_cluster, title="Light GBM (lipiec 2023) - model z miastami", point_size=3, alpha=0.5)
plot_without_city_lgbm = plot_map(df=df_07_2023_predict_without_city_lgbm, parameter_name='Pustynia_predict', colormap=colormap_cluster, title="Light GBM (lipiec 2023) - model bez miast", point_size=3, alpha=0.5)
output_notebook()
show(row(plot_with_city_lgbm, plot_without_city_lgbm))

model_las_m2_lat_path='/content/drive/MyDrive/BigMess/NASA/Modele/Klasyfikacja/Modele_z_lat/las_m2_lat'
model_las_m2_lat = joblib.load(model_las_m2_lat_path)

df_07_2023_predict_lat = df_07_2023.copy()
df_07_2023_predict_lat['Pustynia_predict'] = model_las_m2_lat.predict(df_07_2023_predict_lat.loc[:,'lat':'SoilM_100_200cm'])
df_07_2023_predict_lat = df_07_2023_predict_lat.sort_values(by=['lon', 'lat'])

gs_las_lat = GridSearchCV(RandomForestClassifier(random_state = 2024), cv = 5, param_grid = params_gs_las, scoring = 'accuracy')
gs_las_lat.fit(X_train_lat_bal, y_train_lat_bal)

las_lat = RandomForestClassifier(random_state = 2024, **gs_las_lat.best_params_)
las_lat.fit(X_train_lat_bal, y_train_lat_bal)

summary_model(las_lat, X_train_lat_bal, y_train_lat_bal, ['0','1'])

show_metrics(las_lat, X_train_lat_bal, y_train_lat_bal)

summary_model(las_lat, X_test_lat, y_test_lat, ['0','1'])

show_metrics(las_lat, X_test_lat, y_test_lat)

df_07_2023_predict_without_city_lat = df_07_2023.copy()
df_07_2023_predict_without_city_lat['Pustynia_predict'] = las_lat.predict(df_07_2023_predict_without_city_lat.loc[:,'lat':'SoilM_100_200cm'])
df_07_2023_predict_without_city_lat.loc[df_07_2023_predict_without_city_lat['city'] == 1, 'Pustynia_predict'] = 0
df_07_2023_predict_without_city_lat= df_07_2023_predict_without_city_lat.sort_values(by=['lon', 'lat'])

plot_with_city_lat  = plot_map(df=df_07_2023_predict_lat, parameter_name='Pustynia_predict', colormap=colormap_cluster, title="Las losowy ze zmienna lat (lipiec 2023) - model z miastami", point_size=3, alpha=0.5)
plot_without_city_lat = plot_map(df=df_07_2023_predict_without_city_lat, parameter_name='Pustynia_predict', colormap=colormap_cluster, title="Las losowy ze zmienna lat (lipiec 2023) - model bez miast", point_size=3, alpha=0.5)
output_notebook()
show(row(plot_with_city_lat, plot_without_city_lat))

NASA_sample_an = geo_anotated.toPandas()

monthly_avg_4_july2023, monthly_avg_4_july2023_an_merge = prepare_agg_data_for_training(nasa_sample_an=NASA_sample_an, year=2023, month=7, n_years=4)
monthly_avg_5_july2023, monthly_avg_5_july2023_an_merge = prepare_agg_data_for_training(nasa_sample_an=NASA_sample_an, year=2023, month=7, n_years=5)

city = spark.sql("""
          SELECT DISTINCT
          lon, lat,
          CASE WHEN City = 'NaN' THEN 0 ELSE 1 END AS city
          FROM nasa_city
          """).toPandas()

monthly_avg_4_july2023_city01 = monthly_avg_4_july2023.merge(city, left_on=['lon','lat'], right_on=['lon','lat'], how='inner')

monthly_avg_4_july2023_an_merge_city01 = monthly_avg_4_july2023_an_merge.merge(city, left_on=['lon','lat'], right_on=['lon','lat'], how='inner')
monthly_avg_4_july2023_an_merge_city = monthly_avg_4_july2023_an_merge_city01[monthly_avg_4_july2023_an_merge_city01['city'] == 0]


monthly_avg_5_july2023_city01 = monthly_avg_5_july2023.merge(city, left_on=['lon','lat'], right_on=['lon','lat'], how='inner')

monthly_avg_5_july2023_an_merge_city01 = monthly_avg_5_july2023_an_merge.merge(city, left_on=['lon','lat'], right_on=['lon','lat'], how='inner')
monthly_avg_5_july2023_an_merge_city = monthly_avg_5_july2023_an_merge_city01[monthly_avg_5_july2023_an_merge_city01['city'] == 0]

best_model_rf_path='/content/drive/MyDrive/BigMess/NASA/Modele/Klasyfikacja/Modele_kilkuletnia_agregacja/random_forest_5years_avg_july2023'
best_model_rf = joblib.load(best_model_rf_path)

monthly_avg_5_july2023_pred = monthly_avg_5_july2023.copy()
monthly_avg_5_july2023_pred['pustynia'] = best_model_rf.predict(monthly_avg_5_july2023.loc[:, [col for col in monthly_avg_5_july2023.columns if col.startswith('monthly_avg')]])

best_rf = RandomForestClassifier(max_depth=8, min_samples_leaf=4, n_estimators=70, random_state=3)
X = monthly_avg_5_july2023_an_merge_city.drop(columns=['lon', 'lat', 'pustynia', 'city'])
y = monthly_avg_5_july2023_an_merge_city['pustynia']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=992)

rf_monthly_avg_5_july2023_city = deepcopy(best_rf)
rf_monthly_avg_5_july2023_city.fit(X_train, y_train)

summary_model(rf_monthly_avg_5_july2023_city, X_train, y_train, ['0','1'])

show_metrics(rf_monthly_avg_5_july2023_city, X_train, y_train)

summary_model(rf_monthly_avg_5_july2023_city, X_test, y_test, ['0', '1'])

show_metrics(rf_monthly_avg_5_july2023_city, X_test, y_test)

monthly_avg_5_july2023_pred_city = monthly_avg_5_july2023_city01.copy()
monthly_avg_5_july2023_pred_city['pustynia'] = rf_monthly_avg_5_july2023_city.predict(monthly_avg_5_july2023.loc[:, [col for col in monthly_avg_5_july2023.columns if col.startswith('monthly_avg')]])
monthly_avg_5_july2023_pred_city.loc[monthly_avg_5_july2023_pred_city['city'] == 1, 'pustynia'] = 0

output_notebook()
plot_with_city = plot_map(df=monthly_avg_5_july2023_pred, parameter_name='pustynia',
              colormap=colormap_cluster,
              title='Pustynie (1) i niepustynie (0) - las losowy (lipiec 2023, średnia 5-letnia) - model z miastami',
              point_size=3, alpha=0.5)

plot_without_city = plot_map(df=monthly_avg_5_july2023_pred_city, parameter_name='pustynia',
              colormap=colormap_cluster,
              title='Pustynie (1) i niepustynie (0) - las losowy (lipiec 2023, średnia 5-letnia) - model bez miast',
              point_size=3, alpha=0.5)
show(row(plot_with_city, plot_without_city))

best_model_lgbm_path='/content/drive/MyDrive/BigMess/NASA/Modele/Klasyfikacja/Modele_kilkuletnia_agregacja/lgbm_4years_avg_july2023'
best_model_lgbm = joblib.load(best_model_lgbm_path)

monthly_avg_4_july2023_pred_lgbm = monthly_avg_4_july2023.copy()
monthly_avg_4_july2023_pred_lgbm['pustynia'] = best_model_lgbm.predict(monthly_avg_4_july2023.loc[:, [col for col in monthly_avg_4_july2023.columns if col.startswith('monthly_avg')]])

best_lgbm = LGBMClassifier(max_depth=10, n_estimators=120, num_leaves=30, random_state=3)
X = monthly_avg_4_july2023_an_merge_city.drop(columns=['lon', 'lat', 'pustynia', 'city'])
y = monthly_avg_4_july2023_an_merge_city['pustynia']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=992)

lgbm_monthly_avg_4_july2023_city = deepcopy(best_lgbm)
lgbm_monthly_avg_4_july2023_city.fit(X_train, y_train)

summary_model(lgbm_monthly_avg_4_july2023_city, X_train, y_train, ['0','1'])

show_metrics(lgbm_monthly_avg_4_july2023_city, X_train, y_train)

summary_model(lgbm_monthly_avg_4_july2023_city, X_test, y_test, ['0', '1'])

show_metrics(lgbm_monthly_avg_4_july2023_city, X_test, y_test)

monthly_avg_4_july2023_pred_city = monthly_avg_4_july2023_city01.copy()
monthly_avg_4_july2023_pred_city['pustynia'] = lgbm_monthly_avg_4_july2023_city.predict(monthly_avg_4_july2023.loc[:, [col for col in monthly_avg_4_july2023.columns if col.startswith('monthly_avg')]])
monthly_avg_4_july2023_pred_city.loc[monthly_avg_4_july2023_pred_city['city'] == 1, 'pustynia'] = 0

output_notebook()
plot_with_city = plot_map(df=monthly_avg_4_july2023_pred_lgbm, parameter_name='pustynia',
              colormap=colormap_cluster,
              title='Pustynie (1) i niepustynie (0) - LGBM (lipiec 2023, średnia 4-letnia) - model z miastami',
              point_size=3, alpha=0.5)

plot_without_city = plot_map(df=monthly_avg_4_july2023_pred_city, parameter_name='pustynia',
              colormap=colormap_cluster,
              title='Pustynie (1) i niepustynie (0) - LGBM (lipiec 2023, średnia 4-letnia) - model bez miast',
              point_size=3, alpha=0.5)
show(row(plot_with_city, plot_without_city))
