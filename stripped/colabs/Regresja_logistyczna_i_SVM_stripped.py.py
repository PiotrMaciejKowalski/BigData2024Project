#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('apt-get install openjdk-8-jdk-headless -qq > /dev/null')
get_ipython().system('wget -q dlcdn.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz')
get_ipython().system('tar xf spark-3.5.0-bin-hadoop3.tgz')
get_ipython().system('pip install -q findspark')


# In[ ]:


get_ipython().system('pip install datashader')


get_ipython().system('pip install holoviews hvplot colorcet')


get_ipython().system('pip install geoviews')


# In[ ]:


import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.5.0-bin-hadoop3"

import findspark
findspark.init()


# In[ ]:


get_ipython().system('git clone https://github.com/PiotrMaciejKowalski/BigData2024Project.git')
get_ipython().run_line_magic('cd', 'BigData2024Project')


# In[ ]:


get_ipython().system('chmod 755 /content/BigData2024Project/src/setup.sh')
get_ipython().system('/content/BigData2024Project/src/setup.sh')


# In[ ]:


from pyspark.sql import SparkSession
spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()


# In[ ]:


import sys
sys.path.append('/content/BigData2024Project/src')


# In[ ]:


from typing import List, Tuple, Optional
from itertools import product
import copy
import math
import random
import numpy as np
import matplotlib as mpl
import pandas as pd
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statistics import mean
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, jaccard_score, recall_score
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.base import BaseEstimator
import datashader as ds
import datashader.transfer_functions as tf
import colorcet as cc
import holoviews as hv
import hvplot.pandas
from holoviews.operation.datashader import datashade
import geoviews as gv
import geoviews.tile_sources as gts
from holoviews import opts
from IPython.display import IFrame
from IPython.core.display import display
from bokeh.plotting import show, output_notebook

from big_mess.loaders import default_loader, preprocessed_loader, load_anotated, save_to_csv


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# ##**Wczytanie danych**

# In[ ]:


get_ipython().run_cell_magic('time', '', 'nasa_full = default_loader(spark)\n')


# In[ ]:


nasa_full.createOrReplaceTempView("nasa_full")


# **Wybieramy do testów miesiące letnie (maj, lipiec, sierpień) i dwa miesiące zimowe (grudzień, styczeń) - w oparciu o analizy wyników modelu drzewa decyzyjnego, który gdy trenowany na danych z tych miesięcy, radził sobie najlepiej**

# In[ ]:


nasa_anotated = load_anotated(spark)
nasa_anotated.createOrReplaceTempView("nasa_anotated")

anotated_coordinates= spark.sql('''SELECT lon, lat, pustynia from nasa_anotated''')
anotated_coordinates.show(5)

nasa_full_2019 = spark.sql('''SELECT * from nasa_full where Year=2019''')
nasa_full_2019.show(5)
nasa_full_2021 = spark.sql('''SELECT * from nasa_full where Year=2021''')
nasa_full_2019.show(5)

nasa_2019_an = nasa_full_2019.join(anotated_coordinates, on=['lon', 'lat'], how='inner').toPandas()
nasa_2021_an = nasa_full_2021.join(anotated_coordinates, on=['lon', 'lat'], how='inner').toPandas()


# In[ ]:


January_2019_DF = pd.DataFrame(data = nasa_2019_an[nasa_2019_an['Month']==1])
January_2019_DF.reset_index(inplace=True, drop=True)

January_2021_DF = pd.DataFrame(data = nasa_2021_an[nasa_2021_an['Month']==1])
January_2021_DF.reset_index(inplace=True, drop=True)

May_2019_DF = pd.DataFrame(data = nasa_2019_an[nasa_2019_an['Month']==5])
May_2019_DF.reset_index(inplace=True, drop=True)

May_2021_DF = pd.DataFrame(data = nasa_2021_an[nasa_2021_an['Month']==5])
May_2021_DF.reset_index(inplace=True, drop=True)

July_2019_DF = pd.DataFrame(data = nasa_2019_an[nasa_2019_an['Month']==7])
July_2019_DF.reset_index(inplace=True, drop=True)

July_2021_DF = pd.DataFrame(data = nasa_2021_an[nasa_2021_an['Month']==7])
July_2021_DF.reset_index(inplace=True, drop=True)

August_2019_DF = pd.DataFrame(data = nasa_2019_an[nasa_2019_an['Month']==8])
August_2019_DF.reset_index(inplace=True, drop=True)

August_2021_DF = pd.DataFrame(data = nasa_2021_an[nasa_2021_an['Month']==8])
August_2021_DF.reset_index(inplace=True, drop=True)

December_2019_DF = pd.DataFrame(data = nasa_2019_an[nasa_2019_an['Month']==12])
December_2019_DF.reset_index(inplace=True, drop=True)

December_2021_DF = pd.DataFrame(data = nasa_2021_an[nasa_2021_an['Month']==12])
December_2021_DF.reset_index(inplace=True, drop=True)


# In[ ]:


#Funkcja do generowania podzialu na foldy do crosswalidacji blokowej (spatial block crossvalidation)
#kod ze sprintu 1 zaadoptowany pod Pandasa


def get_grid(grid_cell_size: float, min_lat: float, max_lat: float,
             min_lon: float, max_lon: float) -> Tuple[List[float], List[float]] :
 ''' grid_cell_size - the grid cell size (the length of the side of
                         a square) in degrees '''

 area = (max_lat-min_lat)*(max_lon-min_lon)
 cells_num = area//(grid_cell_size*grid_cell_size)
 actual_grid_size = math.sqrt(area/cells_num)

 xx=np.arange(min_lon, max_lon, step = actual_grid_size)
 yy=np.arange(min_lat, max_lat, step = actual_grid_size)

 return xx, yy

def block_partition(df: pd.DataFrame, block_size: float, min_lat: float,
                    max_lat: float, min_lon: float, max_lon: float) -> pd.DataFrame :
  ''' block_size: approximate size of a block '''

  blocked_df = copy.deepcopy(df)
  block_number = np.zeros(len(df))
  xx, yy = get_grid(block_size, min_lat, max_lat, min_lon, max_lon)

  block_count = 0
  for y_low, y_high in zip(yy[:-1], yy[1:]):
    for x_low, x_high in zip(xx[:-1], xx[1:]):
      block = df[(x_low<=df['lon']) & (df['lon'] <= x_high) & ( y_low<= df['lat']) & (df['lat'] <= y_high)]
      if len(block)>0:    #checking if block is non-empty
         block_number[list(block.index)] = block_count
         block_count+=1

  blocked_df['block'] = block_number
  return blocked_df


def Kfolds(df: pd.DataFrame, k : int, random_state: Optional[int]) -> List[pd.DataFrame]:
  ''' k - number of folds '''

  blocks_num = int(df['block'].max())
  blocks_list = [i for i in range(blocks_num+1)]
  n = (blocks_num+1)//k
  reminder = len(blocks)%k
  folds=[]

  if random_state:
    random.seed(random_state)

  for i in range(k):
      fold = []
      for j in range(n):
          r = random.randint(0, len(blocks_list)-1)
          fold.append(blocks_list[r])
          blocks_list.remove(blocks_list[r])
      folds.append(fold)

  if reminder!=0:
     for b in blocks_list:
        n_fol = random.randint(0, k-1)
        folds[n_fol].append(b)
        blocks_list.remove(b)

  foldsDF_list = [df[df['block'].isin(fold)] for fold in folds]
  foldsDF_list = [pd.DataFrame(data=fold_data).drop(columns=['block']) for fold_data in foldsDF_list]

  return foldsDF_list


# In[ ]:


min_lat = 25.0625
max_lat = 52.9375
min_lon = -124.9375
max_lon = -67.0625

blocks = block_partition(July_2019_DF, 2.5, min_lat, max_lat, min_lon, max_lon)
folds = Kfolds(blocks, 6, random_state=333)


# In[ ]:


for f in folds:
  print(len(f))
  print(len(f[f['pustynia']==1]))          #dla random_state=333 otrzymujemy podział z rozsadna iloscia pustyn w kazdym foldzie


# ##**Selekcja cech**

# ### **Analiza wariancji ANOVA**

# In[ ]:


def test_for_residuals_normality(df: pd.DataFrame, column_name:str, label_name: str='pustynia', labels: List[int]=[1,0]) -> None:

  x = []
  for label in labels:
    r = df[df[label_name]==label][column_name] - df[df[label_name]==label][column_name].mean()
    x += list(r)

  shapiro = stats.shapiro(x)
  print(shapiro)
  sm.qqplot(np.array(x), line='s')

def OLS_report_and_ANOVA(df: pd.DataFrame, group_column_name: str, feature_column_name: str) -> None:

  groups = pd.DataFrame({ group_column_name: df[group_column_name],
                         feature_column_name: df[feature_column_name]})
  model = sm.OLS(groups[feature_column_name], groups[group_column_name]).fit()
  print(model.summary())

  groups2 = pd.DataFrame({ 'group': df[group_column_name], 'feature': df[feature_column_name]})
  model2 = ols(formula="feature~group", data=groups2).fit()   #anova_lm requires formula.api.ols instead of api.OLS
  anova = sm.stats.anova_lm(model2, typ=2)
  print(' ')
  print('ANOVA results:')
  print (anova)


def plot_density(df: pd.DataFrame, group_column_name: str, feature_column_name: str, label_for_desert: int=1, sample_size: int=50, random_state: int=13) -> None:

  df_groups = pd.DataFrame({'Desert': df[df[group_column_name]==label_for_desert][feature_column_name],
                 'Non-desert': df[df[group_column_name]!=label_for_desert][feature_column_name]})
  random.seed(random_state)
  desert_sample = random.sample(list(df_groups['Desert'].dropna()), sample_size)
  nondesert_sample = random.sample(list(df_groups['Non-desert'].dropna()), sample_size)

  samples = pd.DataFrame({'Desert': desert_sample, 'Non-desert': nondesert_sample})
  samples.plot.density()


def perform_Kruskal_Wallis(df: pd.DataFrame, group_column_name:str, feature_column_name: str, label_for_desert: int=1, sample_size: int=100, random_state: int=13) -> None:

  df_groups = pd.DataFrame({'Desert': df[df[group_column_name]==label_for_desert][feature_column_name],
                 'Non-desert': df[df[group_column_name]!=label_for_desert][feature_column_name]})
  random.seed(random_state)
  desert_sample = random.sample(list(df_groups['Desert'].dropna()), sample_size)
  nondes_sample = random.sample(list(df_groups['Non-desert'].dropna()), sample_size)

  print(stats.kruskal(desert_sample, nondes_sample))


# * **DANE LETNIE (lipiec):**

# In[ ]:


summer_data = July_2021_DF.drop(columns=['Month', 'Year'])
summer_data['P/PET'] = summer_data['Rainf']/summer_data['PotEvap']
summer_data.columns


# **1) 'Rainf'**

# In[ ]:


test_for_residuals_normality(summer_data, 'Rainf')


# In[ ]:


plot_density(summer_data, 'pustynia', 'Rainf', random_state=13)


# In[ ]:


OLS_report_and_ANOVA(summer_data, 'pustynia', 'Rainf')


# In[ ]:


perform_Kruskal_Wallis(summer_data, 'pustynia', 'Rainf', random_state=13)


# **2) 'Evap'**

# In[ ]:


test_for_residuals_normality(summer_data, 'Evap')


# In[ ]:


plot_density(summer_data, 'pustynia', 'Evap', random_state=13)


# In[ ]:


OLS_report_and_ANOVA(summer_data, 'pustynia', 'Evap')


# In[ ]:


perform_Kruskal_Wallis(summer_data, 'pustynia', 'Evap', random_state=13)


# **3) 'AvgSurfT'**
# 

# In[ ]:


test_for_residuals_normality(summer_data, 'AvgSurfT')


# In[ ]:


plot_density(summer_data, 'pustynia', 'AvgSurfT', random_state=13)


# In[ ]:


OLS_report_and_ANOVA(summer_data, 'pustynia', 'AvgSurfT')


# In[ ]:


perform_Kruskal_Wallis(summer_data, 'pustynia', 'AvgSurfT', random_state=13)


# **4) 'Albedo'**

# In[ ]:


test_for_residuals_normality(summer_data, 'Albedo')


# In[ ]:


plot_density(summer_data, 'pustynia', 'Albedo', random_state=13)


# In[ ]:


OLS_report_and_ANOVA(summer_data, 'pustynia', 'Albedo')


# In[ ]:


perform_Kruskal_Wallis(summer_data, 'pustynia', 'Albedo', random_state=13)


# **5) 'SoilT_40_100cm'**

# In[ ]:


test_for_residuals_normality(summer_data, 'SoilT_40_100cm')


# In[ ]:


plot_density(summer_data, 'pustynia', 'SoilT_40_100cm')


# In[ ]:


OLS_report_and_ANOVA(summer_data, 'pustynia', 'SoilT_40_100cm')


# In[ ]:


perform_Kruskal_Wallis(summer_data, 'pustynia', 'SoilT_40_100cm', random_state=13)


# **6) 'GVEG'**

# In[ ]:


test_for_residuals_normality(summer_data, 'GVEG')


# In[ ]:


plot_density(summer_data, 'pustynia', 'GVEG', random_state=13)


# In[ ]:


OLS_report_and_ANOVA(summer_data, 'pustynia', 'GVEG')


# In[ ]:


perform_Kruskal_Wallis(summer_data, 'pustynia', 'GVEG', random_state=13)


# **7) 'PotEvap'**

# In[ ]:


test_for_residuals_normality(summer_data, 'PotEvap')


# In[ ]:


plot_density(summer_data, 'pustynia', 'PotEvap', random_state=13)


# In[ ]:


OLS_report_and_ANOVA(summer_data, 'pustynia', 'PotEvap')


# In[ ]:


perform_Kruskal_Wallis(summer_data, 'pustynia', 'PotEvap', random_state=13)


# **8) 'RootMoist'**

# In[ ]:


test_for_residuals_normality(summer_data, 'RootMoist')


# In[ ]:


plot_density(summer_data, 'pustynia', 'RootMoist', random_state=13)


# In[ ]:


OLS_report_and_ANOVA(summer_data, 'pustynia', 'RootMoist')


# In[ ]:


perform_Kruskal_Wallis(summer_data, 'pustynia','RootMoist', random_state=13)


# **9) 'SoilM_100_200cm'**

# In[ ]:


test_for_residuals_normality(summer_data, 'SoilM_100_200cm')


# In[ ]:


plot_density(summer_data, 'pustynia', 'SoilM_100_200cm', random_state=13)


# In[ ]:


OLS_report_and_ANOVA(summer_data, 'pustynia', 'SoilM_100_200cm')


# In[ ]:


perform_Kruskal_Wallis(summer_data, 'pustynia', 'SoilM_100_200cm', random_state=13)


# **10) Nowa proponowana zmienna: 'Rainf'/'PotEvap'**

# In[ ]:


test_for_residuals_normality(summer_data, 'P/PET')


# In[ ]:


plot_density(summer_data, 'pustynia', 'P/PET', random_state=13)


# In[ ]:


OLS_report_and_ANOVA(summer_data, 'pustynia', 'P/PET')


# In[ ]:


perform_Kruskal_Wallis(summer_data, 'pustynia', 'P/PET', random_state=13)


# * **DANE ZIMOWE (grudzień):**

# In[ ]:


winter_data = December_2021_DF.drop(columns=['Month', 'Year'])


# **1) 'Rainf'**

# In[ ]:


test_for_residuals_normality(winter_data, 'Rainf')


# In[ ]:


plot_density(winter_data, 'pustynia', 'Rainf', random_state=13)


# In[ ]:


OLS_report_and_ANOVA(winter_data, 'pustynia', 'Rainf')


# In[ ]:


perform_Kruskal_Wallis(winter_data, 'pustynia', 'Rainf', random_state=13)


# **2) 'Evap'**

# In[ ]:


test_for_residuals_normality(winter_data, 'Evap')


# In[ ]:


plot_density(winter_data, 'pustynia', 'Evap', random_state=13)


# In[ ]:


OLS_report_and_ANOVA(winter_data, 'pustynia','Evap')


# In[ ]:


perform_Kruskal_Wallis(winter_data, 'pustynia','Evap', random_state=13)


# **3) 'AvgSurfT'**

# In[ ]:


test_for_residuals_normality(winter_data, 'AvgSurfT')


# In[ ]:


plot_density(winter_data, 'pustynia', 'AvgSurfT', random_state=13)


# In[ ]:


OLS_report_and_ANOVA(winter_data, 'pustynia', 'AvgSurfT')


# In[ ]:


perform_Kruskal_Wallis(winter_data, 'pustynia', 'AvgSurfT', random_state=13)


# **4) 'Albedo'**

# In[ ]:


test_for_residuals_normality(winter_data, 'Albedo')


# In[ ]:


plot_density(winter_data, 'pustynia', 'Albedo')


# In[ ]:


OLS_report_and_ANOVA(winter_data, 'pustynia', 'Albedo')


# In[ ]:


perform_Kruskal_Wallis(winter_data, 'pustynia', 'Albedo', random_state=13)


# **5) 'SoilT_40_100cm'**

# In[ ]:


test_for_residuals_normality(winter_data, 'SoilT_40_100cm')


# In[ ]:


plot_density(winter_data, 'pustynia', 'SoilT_40_100cm', random_state=13)


# In[ ]:


OLS_report_and_ANOVA(winter_data, 'pustynia', 'SoilT_40_100cm')


# In[ ]:


perform_Kruskal_Wallis(winter_data, 'pustynia', 'SoilT_40_100cm', random_state=13)


# **6) 'GVEG'**

# In[ ]:


test_for_residuals_normality(winter_data, 'GVEG')


# In[ ]:


plot_density(winter_data, 'pustynia', 'GVEG', random_state=13)


# In[ ]:


OLS_report_and_ANOVA(winter_data, 'pustynia', 'GVEG')


# In[ ]:


perform_Kruskal_Wallis(winter_data, 'pustynia', 'GVEG', random_state=13)


# **7) 'PotEvap'**

# In[ ]:


test_for_residuals_normality(winter_data, 'PotEvap')


# In[ ]:


plot_density(winter_data, 'pustynia', 'PotEvap', random_state=13)


# In[ ]:


OLS_report_and_ANOVA(winter_data, 'pustynia', 'PotEvap')


# In[ ]:


perform_Kruskal_Wallis(winter_data, 'pustynia', 'PotEvap', random_state=13)


# **8) 'RootMoist'**

# In[ ]:


test_for_residuals_normality(winter_data, 'RootMoist')


# In[ ]:


plot_density(winter_data, 'pustynia', 'RootMoist', random_state=13)


# In[ ]:


OLS_report_and_ANOVA(winter_data, 'pustynia', 'RootMoist')


# In[ ]:


perform_Kruskal_Wallis(winter_data, 'pustynia', 'RootMoist', random_state=13)


# **9) 'SoilM_100_200cm'**

# In[ ]:


test_for_residuals_normality(winter_data, 'SoilM_100_200cm')


# In[ ]:


plot_density(winter_data, 'pustynia', 'SoilM_100_200cm', random_state=13)


# In[ ]:


OLS_report_and_ANOVA(winter_data, 'pustynia', 'SoilM_100_200cm')


# In[ ]:


perform_Kruskal_Wallis(winter_data, 'pustynia', 'SoilM_100_200cm', random_state=13)


# ###**Analiza korelacji**

# * **DANE LETNIE (lipiec):**

# In[ ]:


full_summer_data = spark.sql('''SELECT * from nasa_full WHERE Month = 7 AND (Year = 2021 OR Year = 2019)''').drop('Month', 'Year', 'lon', 'lat').toPandas()


# In[ ]:


corr = full_summer_data.corr(method = "pearson")
cmap = sns.diverging_palette(250, 323, 80, 60, center='dark', as_cmap=True)
sns.heatmap(corr, vmax =1, vmin=-0.3, cmap = cmap, square = True, linewidths = 0.2, annot=True)


# In[ ]:


sns.pairplot(data=full_summer_data, diag_kind='kde')


# * **DANE ZIMOWE (grudzień):**

# In[ ]:


full_winter_data = spark.sql('''SELECT * from nasa_full WHERE Month = 12 AND (Year = 2021 OR Year = 2019)''').drop('Month', 'Year', 'lon', 'lat').toPandas()


# In[ ]:


corr = full_winter_data.corr(method = "pearson")
cmap = sns.diverging_palette(250, 323, 80, 60, center='dark', as_cmap=True)
sns.heatmap(corr, vmax =1, vmin=-0.3, cmap = cmap, square = True, linewidths = 0.2, annot=True)


# In[ ]:


sns.pairplot(data=full_winter_data, diag_kind='kde')


# **Wnioski**:
# * zarówno dla danych letnich jak i zimowych występuje silna korelacja liniowa pomiędzy zmienną **AvgSurfT** i **SoilT_40_100cm** (współczynnik korelacji Pearsona równy **0.91** w przypadku danych letnich i **0.92** w przypadku danych zimowych + patrz wykres)
# * Zmienna **AvgSurfT** jest **ZIMĄ** dość mocno skorelowana z **PotEvap** (wsp. Pearsona = **0.83**)
# * Zmienna **SoilT_40_100cm** jest **ZIMĄ** dość mocno skorelowana z **PotEvap** (wsp. Pearsona = **0.88**)
# 
# * Testy ANOVA i Kruskala-Wallisa dają w niektórych przypadkach niestabilne wyniki (dla różnych wartości seedów otrzymujemy p_value< 0.05, a czasem powyżej 0.05 dla niektórych analizowanych zmiennych - np. dla Albedo dla danych zimowych), w dodatku nie są spełnione założenia testu ANOVA - rezyduły (residuals) nie mają rozkładu normalnego (wedle testu Shapiro-Wilka) - dlatego wykonano dodatkowo test Kruskala-Wallisa.
# 
# * Pomimo wyżej wymienionych ograniczeń związanych z interpretowaniem wyników testów wydaje się zasadnym wyciągnięcie wniosku o braku istnienia statystycznie istotnej różnicy pomiędzy grupami dla parametru **Rainf** w sezonie **zimowym**.

# # **REGRESJA LOGISTYCZNA**

# In[ ]:


def Kfolds_crossvalidation(model: BaseEstimator, folds: np.ndarray, folds_labels: List[List[int]]) -> dict:
 ''' The function performs Kfolds crossvalidation of a given model
     with given folds '''
 validation_accuracy = []
 validation_recall = []
 validation_precision = []
 validation_ROCAUC = []
 validation_jaccard = []

 k=len(folds)
 for i in range(k):
     X_test = folds[i]
     y_test = folds_labels[i]
     train_folds = [folds[j] for j in range(folds) if j!=i]
     train_labels = [folds_labels[j] for j in range(folds_labels) if j!=i]
     X_train = np.concatenate(train_folds)
     y_train = np.concatenate(train_labels)
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     validation_accuracy.append(accuracy_score(y_test, y_pred))
     validation_recall.append(recall_score(y_test, y_pred))
     validation_precision.append(precision_score(y_test, y_pred, zero_division=0))
     validation_ROCAUC.append(roc_auc_score(y_test, y_pred))
     validation_jaccard.append(jaccard_score(y_test, y_pred))

 results = {'accuracy': mean(validation_accuracy),
            'recall': mean(validation_recall),
            'precision': mean(validation_precision),
            'ROC-AUC': mean(validation_ROCAUC),
            'jaccard_score': mean(validation_jaccard)}

 return results


# In[ ]:


def data_standarization(df: pd.DataFrame) -> np.ndarray:
    ''' the function performs data standarization
        on data stored in given pandas DataFrame '''

    scaler = preprocessing.StandardScaler()
    standarized = scaler.fit_transform(df)

    return standarized


# In[ ]:


def GridSearchCV_LogisticRegression(folds: np.ndarray, folds_labels: List[List[int]], param_grid: dict) -> pd.DataFrame:
  all_res = {'penalty': [], 'solver': [], 'class_weight': [], 'C': [], 'accuracy': [], 'recall': [],
              'precision': [], 'ROC-AUC': [], 'jaccard_score': []}

  param_values = [
    param_grid['solver'],
    param_grid['penalty'],
    param_grid['C'],
    param_grid['class_weight'],
  ]

  for solver, penalty, c, class_weight in product(*param_values):
      model = LogisticRegression(penalty=penalty, class_weight = class_weight, solver=solver, C=c, max_iter=10000)
      results = {'penalty': penalty, 'solver': solver, 'class_weight': str(class_weight), 'C': c}
      crossval_results = Kfolds_crossvalidation(model, folds, folds_labels)
      results.update(crossval_results)
      results_sorted = {key : results[key] for key in all_res.keys()}
      all_res = {key: all_res[key] + [results_sorted[key]] for key in all_res}

  return pd.DataFrame(data=all_res)


# 
# "Our work has shown that regularization is effective at minimizing accuracy differences between feature scaling schema such that the choice of scaling isn’t as critical as a non-regularized model. Despite the bias control effect of regularization, the predictive performance results indicate that standardization is a fit and normalization is a misfit for logistic regression."
# (https://towardsdatascience.com/logistic-regression-and-the-feature-scaling-ensemble-e78a56fc6c1)
# 
# Zgodnie z powyższą konkluzją zastosujemy standaryzację danych.

#  ## **Modele na danych LETNICH**

# ### **Model I:**

#  **cechy**:
# * 'Rainf'
# * 'Evap'
# * 'AvgSurfT'
# * 'Albedo'
# * 'GVEG'
# * 'PotEvap'
# * 'RootMoist'
# * 'SoilM_100_200cm'   
#   usuwamy **'SoilT_40_100cm'**
# 
# 
# 
# 

# **Dane z lipca**

# In[ ]:


#Splitting into folds
blocks = block_partition(July_2019_DF, 2.5, min_lat, max_lat, min_lon, max_lon)
folds_July_2019 = Kfolds(blocks, 6, random_state=333)
folds_07_2019_labels = [list(fold['pustynia']) for fold in folds_July_2019]


# In[ ]:


#Data standarization
folds_July_2019 = [fold.drop(columns=['lon', 'lat', 'Year', 'Month','pustynia','SoilT_40_100cm'])
                   for fold in folds_July_2019]
folds_07_2019_stand = [data_standarization(fold) for fold in folds_July_2019]


# In[ ]:


param_grid = {'penalty': ['l2'],
              'class_weight': [None, 'balanced', {1:2, 0:1}, {1:1.5, 0:1}],
              'solver': ['sag', 'saga', 'newton-cholesky', 'lbfgs'],
              'C':[200, 100, 20, 10, 1.5, 1, 0.8, 0.5, 0.1, 0.01]}

param_grid2 = {'penalty': ['l1'],
              'class_weight': [None, 'balanced', {1:2, 0:1}, {1:1.5, 0:1}],
              'solver': ['liblinear', 'saga'],
              'C':[200, 100, 20, 10, 1.5, 1, 0.8, 0.5, 0.1, 0.01]}


# In[ ]:


results = GridSearchCV_LogisticRegression(folds_07_2019_stand, folds_07_2019_labels, param_grid)


# In[ ]:


results2 = GridSearchCV_LogisticRegression(folds_07_2019_stand, folds_07_2019_labels, param_grid2)


# In[ ]:


all_results = pd.concat([results, results2], ignore_index=True)
all_results.sort_values(by=['accuracy','precision','jaccard_score'], ascending=False).head(25)


# **Dane z sierpnia**

# In[ ]:


#Splitting into folds
blocks = block_partition(August_2019_DF, 2.5, min_lat, max_lat, min_lon, max_lon)
folds_August_2019 = Kfolds(blocks, 6, random_state=333)
folds_08_2019_labels = [list(fold['pustynia']) for fold in folds_August_2019]


# In[ ]:


#Data standarization
folds_August_2019 = [fold.drop(columns=['lon', 'lat', 'Year', 'Month','pustynia','SoilT_40_100cm'])
                   for fold in folds_August_2019]
folds_08_2019_stand = [data_standarization(fold) for fold in folds_August_2019]


# In[ ]:


results = GridSearchCV_LogisticRegression(folds_08_2019_stand, folds_08_2019_labels, param_grid)
results2 = GridSearchCV_LogisticRegression(folds_08_2019_stand, folds_08_2019_labels, param_grid2)


# In[ ]:


allresults= pd.concat([results, results2], ignore_index=True)


# In[ ]:


allresults.sort_values(by=['accuracy', 'precision','jaccard_score'], ascending=False).head(20)


# In[ ]:


allresults.sort_values(by='precision', ascending=False)


# **Dane z maja**

# In[ ]:


#Splitting into folds
blocks = block_partition(May_2019_DF, 2.5, min_lat, max_lat, min_lon, max_lon)
folds_May_2019 = Kfolds(blocks, 6, random_state=333)
folds_05_2019_labels = [list(fold['pustynia']) for fold in folds_May_2019]


# In[ ]:


#Data standarization
folds_May_2019 = [fold.drop(columns=['lon', 'lat', 'Year', 'Month','pustynia','SoilT_40_100cm'])
                   for fold in folds_May_2019]
folds_05_2019_stand = [data_standarization(fold) for fold in folds_May_2019]


# In[ ]:


results = GridSearchCV_LogisticRegression(folds_05_2019_stand, folds_05_2019_labels, param_grid)


# In[ ]:


results2 = GridSearchCV_LogisticRegression(folds_05_2019_stand, folds_05_2019_labels, param_grid2)


# In[ ]:


all_results = pd.concat([results, results2], ignore_index=True)


# In[ ]:


all_results.sort_values(by=['accuracy', 'precision', 'jaccard_score'], ascending=False)


# ## **Model II:**
# 

# **cechy** z Modelu I plus nowa cecha: Rainf/PotEvap (inspirowana aridity index)
# 
# **cechy**:
# * 'Rainf'
# * 'Evap'
# * 'AvgSurf'
# * 'Albedo'
# * 'GVEG'
# * 'RootMoist'
# * 'SoilM_100_200cm'
# * **'Rainf'/'PotEvap'**

# **Dane z lipca**

# In[ ]:


July_2019_DF2 = copy.deepcopy(July_2019_DF)
July_2019_DF2['Rainf/PotEvap'] = (July_2019_DF2['Rainf']/July_2019_DF2['PotEvap'])
#Splitting into folds
blocks = block_partition(July_2019_DF2, 2.5, min_lat, max_lat, min_lon, max_lon)
folds_July_2019 = Kfolds(blocks, 6, random_state=333)
folds_07_2019_labels = [list(fold['pustynia']) for fold in folds_July_2019]


# In[ ]:


#Data standarization
folds_July_2019 = [ fold.drop(columns=['lon', 'lat', 'Year', 'Month','pustynia','SoilT_40_100cm', 'PotEvap'])
                   for fold in folds_July_2019]
folds_07_2019_stand2 = [data_standarization(fold) for fold in folds_July_2019]


# In[ ]:


results = GridSearchCV_LogisticRegression(folds_07_2019_stand2, folds_07_2019_labels, param_grid)


# In[ ]:


results2 = GridSearchCV_LogisticRegression(folds_07_2019_stand2, folds_07_2019_labels, param_grid2)


# In[ ]:


all_results = pd.concat([results, results2], ignore_index=True)
all_results.sort_values(by=['accuracy','precision','recall'], ascending=False)


# **Dane z maja**

# In[ ]:


May_2019_DF2 = copy.deepcopy(May_2019_DF)
May_2019_DF2['Rainf/PotEvap'] = (May_2019_DF2['Rainf']/May_2019_DF2['PotEvap'])
#Splitting into folds
blocks = block_partition(May_2019_DF2, 2.5, min_lat, max_lat, min_lon, max_lon)
folds_May_2019 = Kfolds(blocks, 6, random_state=333)
folds_05_2019_labels = [list(fold['pustynia']) for fold in folds_May_2019]

#Data standarization
folds_May_2019 = [fold.drop(columns=['lon', 'lat', 'Year', 'Month','pustynia','AvgSurfT'])
                   for fold in folds_May_2019]
folds_05_2019_stand2 = [data_standarization(fold) for fold in folds_May_2019]


# In[ ]:


results = GridSearchCV_LogisticRegression(folds_05_2019_stand2, folds_05_2019_labels, param_grid)


# In[ ]:


results2 = GridSearchCV_LogisticRegression(folds_05_2019_stand2, folds_05_2019_labels, param_grid2)


# In[ ]:


all_results = pd.concat([results, results2], ignore_index=True)
all_results.sort_values(by=['accuracy','precision','recall'], ascending=False)


# **Dane z sierpnia**

# In[ ]:


August_2019_DF2 = copy.deepcopy(August_2019_DF)
August_2019_DF2['Rainf/PotEvap'] = (August_2019_DF2['Rainf']/August_2019_DF2['PotEvap'])
#Splitting into folds
blocks = block_partition(August_2019_DF2, 2.5, min_lat, max_lat, min_lon, max_lon)
folds_August_2019 = Kfolds(blocks, 6, random_state=333)
folds_08_2019_labels = [list(fold['pustynia']) for fold in folds_August_2019]

#Data standarization
folds_August_2019 = [fold.drop(columns=['lon', 'lat', 'Year', 'Month','pustynia','AvgSurfT'])
                   for fold in folds_August_2019]
folds_08_2019_stand2 = [data_standarization(fold) for fold in folds_August_2019]


# In[ ]:


results = GridSearchCV_LogisticRegression(folds_08_2019_stand2, folds_08_2019_labels, param_grid)


# In[ ]:


results2 = GridSearchCV_LogisticRegression(folds_08_2019_stand2, folds_08_2019_labels, param_grid2)


# In[ ]:


all_results = pd.concat([results, results2], ignore_index=True)
all_results.sort_values(by=['accuracy','precision','recall'], ascending=False)


# ### **Model III**

# **cechy**:
# * 'Rainf'
# * 'GVEG'
# * 'Evap'
# * 'PotEvap'
# * 'Rainf'/'PotEvap'

# **Dane z lipca**

# In[ ]:


#Splitting into folds
blocks = block_partition(July_2019_DF2, 2.5, min_lat, max_lat, min_lon, max_lon)
folds_July_2019 = Kfolds(blocks, 6, random_state=333)
folds_07_2019_labels = [list(fold['pustynia']) for fold in folds_July_2019]

#Data standarization
folds_July_2019 = [fold.drop(columns=['lon', 'lat', 'Year', 'Month','pustynia','SoilT_40_100cm', 'Albedo', 'AvgSurfT', 'RootMoist', 'SoilM_100_200cm'])
                   for fold in folds_July_2019]
folds_07_2019_stand3 = [data_standarization(fold) for fold in folds_July_2019]


# In[ ]:


results = GridSearchCV_LogisticRegression(folds_07_2019_stand3, folds_07_2019_labels, param_grid)


# In[ ]:


results2 = GridSearchCV_LogisticRegression(folds_07_2019_stand3, folds_07_2019_labels, param_grid2)


# In[ ]:


all_results = pd.concat([results, results2], ignore_index=True)
all_results.sort_values(by=['accuracy','precision','recall'], ascending=False).head(20)


# **Dane z maja**

# In[ ]:


#Splitting into folds
blocks = block_partition(May_2019_DF2, 2.5, min_lat, max_lat, min_lon, max_lon)
folds_May_2019 = Kfolds(blocks, 6, random_state=333)
folds_05_2019_labels = [list(fold['pustynia']) for fold in folds_May_2019]

#Data standarization
folds_May_2019 = [fold.drop(columns=['lon', 'lat', 'Year', 'Month','pustynia','SoilT_40_100cm', 'Albedo', 'AvgSurfT', 'RootMoist', 'SoilM_100_200cm'])
                   for fold in folds_May_2019]
folds_05_2019_stand3 = [data_standarization(fold) for fold in folds_May_2019]


# In[ ]:


results = GridSearchCV_LogisticRegression(folds_05_2019_stand3, folds_05_2019_labels, param_grid)


# In[ ]:


results2 = GridSearchCV_LogisticRegression(folds_05_2019_stand3, folds_05_2019_labels, param_grid2)


# In[ ]:


all_results = pd.concat([results, results2], ignore_index=True)
all_results.sort_values(by=['accuracy','precision','recall'], ascending=False)


# **Dane z sierpnia**

# In[ ]:


#Splitting into folds
blocks = block_partition(August_2019_DF2, 2.5, min_lat, max_lat, min_lon, max_lon)
folds_August_2019 = Kfolds(blocks, 6, random_state=333)
folds_08_2019_labels = [list(fold['pustynia']) for fold in folds_August_2019]

#Data standarization
folds_August_2019 = [fold.drop(columns=['lon', 'lat', 'Year', 'Month','pustynia','SoilT_40_100cm', 'Albedo', 'AvgSurfT', 'RootMoist', 'SoilM_100_200cm'])
                   for fold in folds_August_2019]
folds_08_2019_stand3 = [data_standarization(fold) for fold in folds_August_2019]


# In[ ]:


results = GridSearchCV_LogisticRegression(folds_08_2019_stand3, folds_08_2019_labels, param_grid)


# In[ ]:


results2 = GridSearchCV_LogisticRegression(folds_08_2019_stand3, folds_08_2019_labels, param_grid2)


# In[ ]:


all_results = pd.concat([results, results2], ignore_index=True)
all_results.sort_values(by=['accuracy','precision','recall'], ascending=False)


# ## **Modele na danych ZIMOWYCH**

# ### **Model I**

#  **cechy**:
# * 'Rainf'
# * 'Evap'
# * 'AvgSurfT'
# * 'Albedo'
# * 'GVEG'
# * 'PotEvap'
# * 'RootMoist'
# * 'SoilM_100_200cm'   
#   usuwamy **'SoilT_40_100cm'**
# 

# **Dane z grudnia**

# In[ ]:


#Splitting into folds
blocks = block_partition(December_2019_DF, 2.5, min_lat, max_lat, min_lon, max_lon)
folds_December_2019 = Kfolds(blocks, 6, random_state=333)
folds_12_2019_labels = [list(fold['pustynia']) for fold in folds_December_2019]

#Data standarization
folds_December_2019 = [fold.drop(columns=['lon', 'lat', 'Year', 'Month','pustynia','SoilT_40_100cm'])
                   for fold in folds_December_2019]
folds_12_2019_stand = [data_standarization(fold) for fold in folds_December_2019]


# In[ ]:


results = GridSearchCV_LogisticRegression(folds_12_2019_stand, folds_12_2019_labels, param_grid)


# In[ ]:


results2 = GridSearchCV_LogisticRegression(folds_12_2019_stand, folds_12_2019_labels, param_grid2)


# In[ ]:


all_results = pd.concat([results, results2], ignore_index=True)
all_results.sort_values(by=['accuracy','precision','recall'], ascending=False)


# **Dane ze stycznia**

# In[ ]:


#Splitting into folds
blocks = block_partition(January_2019_DF, 2.5, min_lat, max_lat, min_lon, max_lon)
folds_January_2019 = Kfolds(blocks, 6, random_state=333)
folds_01_2019_labels = [list(fold['pustynia']) for fold in folds_January_2019]

#Data standarization
folds_January_2019 = [fold.drop(columns=['lon', 'lat', 'Year', 'Month','pustynia','SoilT_40_100cm'])
                   for fold in folds_January_2019]
folds_01_2019_stand = [data_standarization(fold) for fold in folds_January_2019]


# In[ ]:


results = GridSearchCV_LogisticRegression(folds_01_2019_stand, folds_01_2019_labels, param_grid)


# In[ ]:


results2 = GridSearchCV_LogisticRegression(folds_01_2019_stand, folds_01_2019_labels, param_grid2)


# In[ ]:


all_results = pd.concat([results, results2], ignore_index=True)
all_results.sort_values(by=['precision','recall'], ascending=False)


# ### **Model II**

#  **cechy**:
# * 'Rainf'
# * 'Evap'
# * 'AvgSurfT'
# * 'Albedo'
# * 'GVEG'
# * 'RootMoist'
# * 'SoilM_100_200cm'   
#   usuwamy **'SoilT_40_100cm'** oraz **'PotEvap'**
# 

# **Dane z grudnia**

# In[ ]:


#Splitting into folds
blocks = block_partition(December_2019_DF, 2.5, min_lat, max_lat, min_lon, max_lon)
folds_December_2019 = Kfolds(blocks, 6, random_state=333)
folds_12_2019_labels = [list(fold['pustynia']) for fold in folds_December_2019]

#Data standarization
folds_December_2019 = [fold.drop(columns=['lon', 'lat', 'Year', 'Month','pustynia','SoilT_40_100cm', 'PotEvap'])
                   for fold in folds_December_2019]
folds_12_2019_stand2 = [data_standarization(fold) for fold in folds_December_2019]


# In[ ]:


results = GridSearchCV_LogisticRegression(folds_12_2019_stand2, folds_12_2019_labels, param_grid)
results2 = GridSearchCV_LogisticRegression(folds_12_2019_stand2, folds_12_2019_labels, param_grid2)


# In[ ]:


all_results = pd.concat([results, results2], ignore_index=True)
all_results.sort_values(by=['precision','recall'], ascending=False)


# ### **Model III**

#  **cechy**:  
# * 'Evap'
# * 'AvgSurfT'
# * 'Albedo'
# * 'GVEG'
# * 'RootMoist'
# * 'SoilM_100_200cm'   
#   usuwamy **'SoilT_40_100cm'** oraz **'PotEvap'** oraz **'Rainf'**

# **Dane z grudnia**

# In[ ]:


#Splitting into blocks:
blocks = block_partition(December_2019_DF, 2.5, min_lat, max_lat, min_lon, max_lon)
folds_December_2019 = Kfolds(blocks, 6, random_state=333)
#Data standarization"
folds_December_2019 = [fold.drop(columns=['lon', 'lat', 'Year', 'Month','pustynia','SoilT_40_100cm','PotEvap', 'Rainf'])
                   for fold in folds_December_2019]
folds_12_2019_stand3= [data_standarization(fold) for fold in folds_December_2019]


# In[ ]:


results = GridSearchCV_LogisticRegression(folds_12_2019_stand3, folds_12_2019_labels, param_grid)
results2 = GridSearchCV_LogisticRegression(folds_12_2019_stand3, folds_12_2019_labels, param_grid2)


# In[ ]:


all_results = pd.concat([results, results2], ignore_index=True)
all_results.sort_values(by=['accuracy','precision', 'recall'], ascending=False).head(20)


# ###**Model IV**

#  **cechy**:  
# * 'Evap'
# * 'AvgSurfT'
# * 'Albedo'
# * 'GVEG'
# * 'PotEvap'
# * 'RootMoist'
# * 'SoilM_100_200cm'   
#   usuwamy **'SoilT_40_100cm'** oraz **'AvgSurfT'** oraz **'Rainf'**

# **Dane z grudnia**

# In[ ]:


#Splitting into folds
blocks = block_partition(December_2019_DF, 2.5, min_lat, max_lat, min_lon, max_lon)
folds_December_2019 = Kfolds(blocks, 6, random_state=333)
folds_12_2019_labels = [list(fold['pustynia']) for fold in folds_December_2019]

#Data standarization
folds_December_2019 = [fold.drop(columns=['lon', 'lat', 'Year', 'Month','pustynia','SoilT_40_100cm', 'AvgSurfT', 'Rainf'])
                   for fold in folds_December_2019]
folds_12_2019_stand4 = [data_standarization(fold) for fold in folds_December_2019]


# In[ ]:


results = GridSearchCV_LogisticRegression(folds_12_2019_stand4, folds_12_2019_labels, param_grid)
results2 = GridSearchCV_LogisticRegression(folds_12_2019_stand4, folds_12_2019_labels, param_grid2)


# In[ ]:


all_results = pd.concat([results, results2], ignore_index=True)
all_results.sort_values(by=['accuracy','precision','recall'], ascending=False)


# ### **Model V**
# 

#  **cechy**:
# * 'Rainf'
# * 'Evap'
# * 'AvgSurfT'
# * 'GVEG'
# * 'PotEvap'
# * 'RootMoist'
# * 'SoilM_100_200cm'   
#   usuwamy **'SoilT_40_100cm'** oraz **'Rainf'** oraz **'Albedo'**

# **Dane z grudnia**

# In[ ]:


#Splitting into folds
blocks = block_partition(December_2019_DF, 2.5, min_lat, max_lat, min_lon, max_lon)
folds_December_2019 = Kfolds(blocks, 6, random_state=333)
folds_12_2019_labels = [list(fold['pustynia']) for fold in folds_December_2019]

#Data standarization
folds_December_2019 = [fold.drop(columns=['lon', 'lat', 'Year', 'Month','pustynia','SoilT_40_100cm', 'Rainf', 'Albedo', 'Evap'])
                   for fold in folds_December_2019]
folds_12_2019_stand5 = [data_standarization(fold) for fold in folds_December_2019]


# In[ ]:


results = GridSearchCV_LogisticRegression(folds_12_2019_stand5, folds_12_2019_labels, param_grid)
results2 = GridSearchCV_LogisticRegression(folds_12_2019_stand5, folds_12_2019_labels, param_grid2)


# In[ ]:


all_results = pd.concat([results, results2], ignore_index=True)
all_results.sort_values(by=['accuracy','precision','recall'], ascending=False)


# ##**Przykładowe wizualizacje wybranych modeli**

# In[ ]:


def get_colormap(values: list, colors_palette: list, name = 'custom'):
    values = np.sort(np.array(values))
    values = np.interp(values, (values.min(), values.max()), (0, 1))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(name, list(zip(values, colors_palette)))
    return cmap

def plot_map(df: pd.DataFrame, parameter_name: str, colormap: mpl.colors.LinearSegmentedColormap,
             point_size: int = 8, width: int = 900, height: int = 600, alpha: float = 1,
             bgcolor: str = 'white', colorbar: Optional[bool] = False):

    gdf = gv.Points(df, ['lon', 'lat'], [parameter_name]) # obiekt zawierający punkty
    tiles = gts.OSM # wybór mapy tła, w tym wypadku OpenStreetMap

    # łączenie mapy tła z punktami i ustawienie wybranych parametrów wizualizacji
    map_with_points = tiles * gdf.opts(
        color=parameter_name,
        cmap=colormap,
        size=point_size,
        width=width,
        height=height,
        colorbar=colorbar,
        toolbar='above',
        tools=['hover', 'wheel_zoom', 'reset'],
        alpha=alpha
    )

    return hv.render(map_with_points)


# In[ ]:


nasa2019full = nasa_full_2019.toPandas()


# In[ ]:


nasa2021full = nasa_full_2021.toPandas()


# In[ ]:


def train_and_predict(model: BaseEstimator, df_train: pd.DataFrame, df_fit: pd.DataFrame, label_column_name: str, proba: bool=False) -> np.ndarray:

  y_train = df_train[label_column_name]
  X_train = data_standarization(df_train.drop(columns=[label_column_name]))
  trained_model = model.fit(X_train, y_train)
  X = data_standarization(df_fit)
  if proba:
    y_pred = trained_model.predict_proba(X)[:,1]
  else:
    y_pred = trained_model.predict(X)

  return y_pred


# * **Model I dla danych z lipca (wytrenowany na danych z lipca 2019) - predykcje dla danych z lipca 2019:**

# * z parametrami: regularyzacja **L2**,  solver =	**sag**, C = 10 ,    weighted_class = {1:2, 0:1}
# 
# 
# 

# In[ ]:


model = LogisticRegression(solver='sag', class_weight={1:2, 0:1}, penalty='l2', C=10, max_iter=10000)
df_train = July_2019_DF.drop(columns=['lon','lat', 'Year', 'Month', 'SoilT_40_100cm'])
nasa_07_2019 = pd.DataFrame(nasa2019full[nasa2019full['Month']==7])
nasa_fit = nasa_07_2019.drop(columns=['lon', 'lat', 'Year', 'Month', 'SoilT_40_100cm'])
y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia')
labels = pd.DataFrame({'lon': nasa_07_2019['lon'], 'lat': nasa_07_2019['lat'], 'label': y_pred})


# In[ ]:


colormap = get_colormap([0, 1], ['darkgreen', 'yellow'])
output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap), alpha=0.5)


# * **Model I dla danych z lipca (wytrenowany na danych z lipca 2019) - predykcje dla danych z lipca 2021:**

# In[ ]:


nasa_07_2021 = pd.DataFrame(nasa2021full[nasa2021full['Month']==7])
nasa_fit = nasa_07_2021.drop(columns=['lon', 'lat','Year', 'Month', 'SoilT_40_100cm'])
y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia')
labels = pd.DataFrame({'lon': nasa_07_2021['lon'], 'lat': nasa_07_2021['lat'], 'label': y_pred})


# In[ ]:


output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap), alpha=0.5)


# * **Mapa prawdopodobieństw:**

# In[ ]:


y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia', proba=True)
labels = pd.DataFrame({'lon': nasa_07_2021['lon'], 'lat': nasa_07_2021['lat'], 'label': y_pred})

output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap, colorbar=True), alpha=0.5)


# * **Model I dla danych z lipca (wytrenowany na danych z lipca 2019) - predykcje dla danych z lipca 2021:**
# 
# * z parametrami: regularyzacja **L1**, solver = **saga**,  class_weight={1:2, 0:1},  C = **0.5**

# In[ ]:


model = LogisticRegression(solver='saga', class_weight={1:2, 0:1}, penalty='l1', C=0.5, max_iter=10000)
df_train = July_2019_DF.drop(columns=['lon','lat', 'Year', 'Month', 'SoilT_40_100cm'])
nasa_07_2021 = pd.DataFrame(nasa2021full[nasa2021full['Month']==7])
nasa_fit = nasa_07_2021.drop(columns=['lon', 'lat', 'Year', 'Month', 'SoilT_40_100cm'])
y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia')
labels = pd.DataFrame({'lon': nasa_07_2021['lon'], 'lat': nasa_07_2021['lat'], 'label': y_pred})


# In[ ]:


output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap), alpha=0.5)


# * **Mapa prawdopodobieństw**

# In[ ]:


y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia', proba=True)
labels = pd.DataFrame({'lon': nasa_07_2021['lon'], 'lat': nasa_07_2021['lat'], 'label': y_pred})

output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap, colorbar=True), alpha=0.5)


# * **Model I trenowany na danych z sierpnia 2019, predykcje dla danych z sierpnia 2021:**
# 
# * z parametrami: regularyzacja **L2**, solver = **saga**, class_weight = {1:2, 0:1},  C =0.1

# In[ ]:


model = LogisticRegression(solver='saga', class_weight={1:2, 0:1}, penalty='l2', C=0.1, max_iter=10000)
df_train = August_2019_DF.drop(columns=['lon','lat', 'Year', 'Month', 'SoilT_40_100cm'])
nasa_08_2021 = pd.DataFrame(nasa2021full[nasa2021full['Month']==8])
nasa_fit = nasa_08_2021.drop(columns=['lon', 'lat', 'Year', 'Month', 'SoilT_40_100cm'])
y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia')
labels = pd.DataFrame({'lon': nasa_08_2021['lon'], 'lat': nasa_08_2021['lat'], 'label': y_pred})


# In[ ]:


output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap), alpha=0.5)


# * **Mapa prawdopodobieństw**

# In[ ]:


y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia', proba=True)
labels = pd.DataFrame({'lon': nasa_08_2021['lon'], 'lat': nasa_08_2021['lat'], 'label': y_pred})

output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap, colorbar=True), alpha=0.5)


# * **Model I trenowany na danych z sierpnia 2019, predykcje dla danych z sierpnia 2021:**
# 
# * z parametrami: regularyzacja **L2**, solver = **sag**,  class_weight = {1:1.5, 0:1},  C =0.1

# In[ ]:


model = LogisticRegression(solver='sag', class_weight={1:1.5, 0:1}, penalty='l2', C=0.1, max_iter=10000)
df_train = August_2019_DF.drop(columns=['lon','lat', 'Year', 'Month', 'SoilT_40_100cm'])
nasa_08_2021 = pd.DataFrame(nasa2021full[nasa2021full['Month']==8])
nasa_fit = nasa_08_2021.drop(columns=['lon', 'lat', 'Year', 'Month', 'SoilT_40_100cm'])
y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia')
labels = pd.DataFrame({'lon': nasa_08_2021['lon'], 'lat': nasa_08_2021['lat'], 'label': y_pred})


# In[ ]:


output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap), alpha=0.5)


# * **Mapa prawdopodobieństw**

# In[ ]:


y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia', proba=True)
labels = pd.DataFrame({'lon': nasa_08_2021['lon'], 'lat': nasa_08_2021['lat'], 'label': y_pred})

output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap, colorbar=True), alpha=0.5)


# * **Model I trenowany na danych z sierpnia 2019, predykcje dla danych z sierpnia 2021:**
# 
# * z parametrami: regularyzacja **L1**, solver = **liblinear**,   class_weight = {1: 1.5, 0:1},    C = 0.5
# 
# 

# In[ ]:


model = LogisticRegression(solver='liblinear', class_weight={1:1.5, 0:1}, penalty='l1', C=0.5, max_iter=10000)
df_train = August_2019_DF.drop(columns=['lon','lat', 'Year', 'Month', 'SoilT_40_100cm'])
nasa_08_2021 = pd.DataFrame(nasa2021full[nasa2021full['Month']==8])
nasa_fit = nasa_08_2021.drop(columns=['lon', 'lat', 'Year', 'Month', 'SoilT_40_100cm'])
y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia')
labels = pd.DataFrame({'lon': nasa_08_2021['lon'], 'lat': nasa_08_2021['lat'], 'label': y_pred})


# In[ ]:


output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap), alpha=0.5)


# * **Mapa prawdopodobieństw**

# In[ ]:


y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia', proba=True)
labels = pd.DataFrame({'lon': nasa_08_2021['lon'], 'lat': nasa_08_2021['lat'], 'label': y_pred})

output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap, colorbar=True), alpha=0.5)


# * **Model II trenowany na danych z lipca 2019, predykcje dla danych z lipca 2021 :**
# 
# * z parametrami: regularyzacja **L2**, solver = **newton-cholesky**,   class_weight = {1:1.5, 0:1},    C = 10
# 

# In[ ]:


model = LogisticRegression(solver='newton-cholesky', class_weight={1:1.5, 0:1}, penalty='l2', C=10, max_iter=10000)
df_train = July_2019_DF.drop(columns=['lon','lat', 'Year', 'Month', 'SoilT_40_100cm'])
df_train['Rainf/Potevap'] =  df_train['Rainf']/df_train['PotEvap']
df_train = df_train.drop(columns=['PotEvap'])
nasa_07_2021 = pd.DataFrame(nasa2021full[nasa2021full['Month']==7])
nasa_fit = nasa_07_2021.drop(columns=['lon', 'lat', 'Year', 'Month', 'SoilT_40_100cm'])
nasa_fit['Rainf/PotEvap'] = nasa_fit['Rainf']/nasa_fit['PotEvap']
nasa_fit = nasa_fit.drop(columns=['PotEvap'])
y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia')
labels = pd.DataFrame({'lon': nasa_07_2021['lon'], 'lat': nasa_07_2021['lat'], 'label': y_pred})


# In[ ]:


output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap), alpha=0.5)


# * **Mapa prawdopodobieństw**

# In[ ]:


y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia', proba=True)
labels = pd.DataFrame({'lon': nasa_07_2021['lon'], 'lat': nasa_07_2021['lat'], 'label': y_pred})

output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap, colorbar=True), alpha=0.5)


# * **Model III trenowany na danych z lipca 2019, predykcje dla danych z lipca 2021:**

# * z parametrami: regularyzacja **L2**,  solver =	**saga**, C = 1.0 ,    weighted_class = {1:1.5, 0:1}

# In[ ]:


model3 = LogisticRegression(solver='saga', class_weight={1:1.5, 0:1}, penalty='l2', C=1, max_iter=10000)
df_train = July_2019_DF.drop(columns=['lon','lat', 'Year', 'Month', 'SoilT_40_100cm', 'AvgSurfT', 'RootMoist', 'SoilM_100_200cm','Albedo'])
df_train['P/PET'] = df_train['Rainf']/df_train['PotEvap']
nasa_07_2021 = pd.DataFrame(nasa2021full[nasa2021full['Month']==7])
nasa_fit = nasa_07_2021.drop(columns=['lon', 'lat', 'Year', 'Month', 'SoilT_40_100cm', 'AvgSurfT', 'RootMoist', 'SoilM_100_200cm','Albedo'])
nasa_fit['P/PET'] = nasa_fit['Rainf']/nasa_fit['PotEvap']
y_pred = train_and_predict(model3, df_train, nasa_fit, 'pustynia')
labels = pd.DataFrame({'lon': nasa_07_2021['lon'], 'lat': nasa_07_2021['lat'], 'label': y_pred})


# In[ ]:


output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap), alpha=0.3)


# * **Mapa prawdopodobieństw**

# In[ ]:


y_pred = train_and_predict(model3, df_train, nasa_fit, 'pustynia', proba=True)
labels = pd.DataFrame({'lon': nasa_07_2021['lon'], 'lat': nasa_07_2021['lat'], 'label': y_pred})

output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap, colorbar=True), alpha=0.5)


# * **Model I trenowany na danych z grudnia 2019, predykcje dla danych z grudnia 2021:**
# 
# * z parametrami: regularyzacja **L1**, solver = **liblinear**,   class_weight = None,    C = 0.5
# 

# In[ ]:


model = LogisticRegression(solver='liblinear', class_weight=None, penalty='l1', C=0.5, max_iter=10000)
df_train = December_2019_DF.drop(columns=['lon','lat', 'Year', 'Month', 'SoilT_40_100cm'])
nasa_12_2021 = pd.DataFrame(nasa2021full[nasa2021full['Month']==12])
nasa_fit = nasa_12_2021.drop(columns=['lon', 'lat', 'Year', 'Month', 'SoilT_40_100cm'])
y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia')
labels = pd.DataFrame({'lon': nasa_12_2021['lon'], 'lat': nasa_12_2021['lat'], 'label': y_pred})


# In[ ]:


output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap), alpha=0.5)


# * **Mapa prawdopoodbieństw**

# In[ ]:


y_pred = train_and_predict(model3, df_train, nasa_fit, 'pustynia', proba=True)
labels = pd.DataFrame({'lon': nasa_12_2021['lon'], 'lat': nasa_12_2021['lat'], 'label': y_pred})

output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap, colorbar=True), alpha=0.5)


# * **Model I trenowany na danych ze stycznia 2019, predykcje dla danych ze stycznia 2021:**
# 
# * z parametrami: regularyzacja **L1**, solver = **liblinear**,   class_weight = {1:2, 0:1},    C = 0.1
# 

# In[ ]:


model = LogisticRegression(solver='liblinear', class_weight={1:2, 0:1}, penalty='l1', C=0.1, max_iter=10000)
df_train = January_2019_DF.drop(columns=['lon', 'lat', 'Year', 'Month','SoilT_40_100cm'])
nasa_01_2021 = pd.DataFrame(nasa2021full[nasa2021full['Month']==1])
nasa_fit = nasa_01_2021.drop(columns=['lon', 'lat', 'Year', 'Month','SoilT_40_100cm'])
y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia')
labels = pd.DataFrame({'lon': nasa_01_2021['lon'], 'lat': nasa_01_2021['lat'], 'label': y_pred})


# In[ ]:


output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap), alpha=0.5)


# * **Model III trenowany na danych z grudnia 2019, predykcje dla danych z grudnia 2021:**
# 
# * z parametrami: regularyzacja **L1**, solver = **liblinear**,   class_weight = {1:2, 0:1},    C = 0.1
# 

# In[ ]:


model = LogisticRegression(solver='liblinear', class_weight={1:2, 0:1}, penalty='l1', C=0.1, max_iter=10000)
df_train = December_2019_DF.drop(columns=['lon', 'lat', 'Year', 'Month','SoilT_40_100cm','Rainf', 'PotEvap'])
nasa_12_2021 = pd.DataFrame(nasa2021full[nasa2021full['Month']==12])
nasa_fit = nasa_12_2021.drop(columns=['lon', 'lat', 'Year', 'Month','SoilT_40_100cm','Rainf', 'PotEvap'])
y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia')
labels = pd.DataFrame({'lon': nasa_12_2021['lon'], 'lat': nasa_12_2021['lat'], 'label': y_pred})


# In[ ]:


output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap), alpha=0.5)


# * **Mapa prawdopodobieństw**

# In[ ]:


y_pred = train_and_predict(model3, df_train, nasa_fit, 'pustynia', proba=True)
labels = pd.DataFrame({'lon': nasa_12_2021['lon'], 'lat': nasa_12_2021['lat'], 'label': y_pred})

output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap, colorbar=True), alpha=0.5)


# # **Maszyny wektorów nośnych - SVM**

# In[ ]:


def GridSearchCV_SVM(folds: np.ndarray, folds_labels: List[List[int]], param_grid: dict, kernel: str) -> pd.DataFrame:

 if kernel == 'linear':
     all_res = {'kernel': [], 'penalty': [], 'C': [], 'class_weight': [], 'loss': [], 'intercept_scal':[],
                'accuracy': [], 'recall': [], 'precision': [], 'ROC-AUC': [], 'jaccard_score': []}
     param_values = [
                      param_grid['loss'],
                      param_grid['penalty'],
                      param_grid['C'],
                      param_grid['class_weight'],
                      param_grid['intercept_scal']
                    ]

     for loss, penalty, C, class_weight, intercept_scal in product(*param_values):
         if (loss =='hinge')&(penalty=='l2'):
                 model = LinearSVC(C=C, penalty=penalty, class_weight= class_weight, loss=loss, dual= True,
                               intercept_scaling = intercept_scal, max_iter = 100000, random_state=13)
         else:
                 model = LinearSVC(C=C, penalty=penalty, class_weight= class_weight, loss=loss, dual= False,
                               intercept_scaling = intercept_scal, max_iter = 10000, random_state=13)
         results = {'kernel': 'linear', 'penalty': penalty, 'C': C, 'class_weight': class_weight,
                    'loss': loss, 'intercept_scal': intercept_scal}
         crossval_results = Kfolds_crossvalidation(model, folds, folds_labels)
         results.update(crossval_results)
         results_sorted = dict({key : results[key] for key in all_res.keys()})
         all_res = {key: all_res[key] + [results_sorted[key]] for key in all_res}

 else:
      param_grid['kernel'] = [kernel]
      all_res = {key : [] for key in param_grid.keys()}
      metrics = {'accuracy': [], 'recall': [], 'precision': [], 'ROC-AUC': [], 'jaccard_score': []}
      all_res.update(metrics)
      hyperparameters = [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]
      for hyperparams in hyperparameters:
         model = SVC(**hyperparams, max_iter=10000)
         results = hyperparams
         results['kernel'] = kernel
         crossval_results = Kfolds_crossvalidation(model, folds, folds_labels)
         results.update(crossval_results)
         results_sorted = dict({key : results[key] for key in all_res.keys()})
         all_res = {key: all_res[key] + [results_sorted[key]] for key in all_res}

 return pd.DataFrame(data=all_res)


#  ## **Linear kernel**

# In[ ]:


param_grid_linear = {'penalty': ['l1', 'l2'],
                     'C': [0.01, 0.1, 0.5, 0.8, 1, 5, 10, 50, 100],
                     'loss': ['squared_hinge'],                           #with loss='hinge' convergence issues occurred
                     'class_weight': [None, 'balanced', {1:1.5, 0:1}],
                     'intercept_scal': [0.1, 1, 1.5, 5, 10, 50]}


# **Dane z lipca**

# In[ ]:


#Splitting into folds
blocks = block_partition(July_2019_DF, 2.5, min_lat, max_lat, min_lon, max_lon)
folds_July_2019 = Kfolds(blocks, 6, random_state=333)
folds_07_2019_labels = [list(fold['pustynia']) for fold in folds_July_2019]

#Data standarization
folds_July_2019 = [fold.drop(columns=['lon', 'lat', 'Year', 'Month','pustynia','SoilT_40_100cm'])
                   for fold in folds_July_2019]
folds_07_2019_stand = [data_standarization(fold) for fold in folds_July_2019]


# In[ ]:


linear_results1 = GridSearchCV_SVM(folds_07_2019_stand, folds_07_2019_labels, param_grid_linear, kernel='linear')
linear_results1.sort_values(by=['accuracy', 'precision'], ascending=False)


# **Dane z sierpnia**

# In[ ]:


#Splitting into folds
blocks = block_partition(August_2019_DF, 2.5, min_lat, max_lat, min_lon, max_lon)
folds_August_2019 = Kfolds(blocks, 6, random_state=333)
folds_08_2019_labels = [list(fold['pustynia']) for fold in folds_August_2019]

#Data standarization
folds_August_2019 = [fold.drop(columns=['lon', 'lat', 'Year', 'Month','pustynia','SoilT_40_100cm'])
                   for fold in folds_August_2019]
folds_08_2019_stand = [data_standarization(fold) for fold in folds_August_2019]


# In[ ]:


linear_results2 = GridSearchCV_SVM(folds_08_2019_stand, folds_08_2019_labels, param_grid_linear, kernel='linear')
linear_results2.sort_values(by=['accuracy', 'precision'], ascending=False)


# **Dane z grudnia**

# In[ ]:


#Splitting into folds
blocks = block_partition(December_2019_DF, 2.5, min_lat, max_lat, min_lon, max_lon)
folds_December_2019 = Kfolds(blocks, 6, random_state=333)
folds_12_2019_labels = [list(fold['pustynia']) for fold in folds_December_2019]

#Data standarization
folds_December_2019 = [fold.drop(columns=['lon', 'lat', 'Year', 'Month','pustynia','SoilT_40_100cm'])
                   for fold in folds_December_2019]
folds_12_2019_stand = [data_standarization(fold) for fold in folds_December_2019]


# In[ ]:


linear_results3 = GridSearchCV_SVM(folds_12_2019_stand, folds_12_2019_labels, param_grid_linear, kernel='linear')
linear_results3.sort_values(by=['accuracy', 'precision'], ascending=False)


# ## **Polynomial kernel**

# In[ ]:


param_grid_poly = {'C': [0.001, 0.01, 0.1, 0.5, 0.8, 1, 10, 50],
                   'class_weight': [None, 'balanced', {1:1.5, 0:1}],
                   'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10, 100],
                   'degree': [2, 3], 'coef0': [0, 0.1, 0.5, 1]}


# **Dane z lipca**

# In[ ]:


poly_results = GridSearchCV_SVM(folds_07_2019_stand, folds_07_2019_labels, param_grid_poly, kernel='poly')  #Zbyt duża złożoność obliczeniowa, by przeliczyć tyle modeli w rozsądnym czasie...


# In[ ]:


poly_results.sort_values(by=['accuracy', 'precision'], ascending=False)


# ## **RBF kernel**

# In[ ]:


param_grid_rbf = {'C': [0.001, 0.01, 0.1, 0.5, 0.8, 1, 1.5, 5, 10, 20, 50],
                  'class_weight': [None, 'balanced', {1:1.5, 0:1}, {1:2, 0:1}],
                  'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10, 100]}


# **Dane z lipca**

# In[ ]:


rbf_results = GridSearchCV_SVM(folds_07_2019_stand, folds_07_2019_labels, param_grid_rbf, kernel='rbf')


# In[ ]:


rbf_results.sort_values(by=['accuracy', 'precision', 'recall'], ascending=False).head(20)


# **Dane z sierpnia**

# In[ ]:


rbf_results2 = GridSearchCV_SVM(folds_08_2019_stand, folds_08_2019_labels, param_grid_rbf, kernel='rbf')


# In[ ]:


rbf_results2.sort_values(by=['accuracy', 'precision', 'recall'], ascending=False).head(20)


# **Dane z grudnia**

# In[ ]:


rbf_results3 = GridSearchCV_SVM(folds_12_2019_stand, folds_12_2019_labels, param_grid_rbf, kernel='rbf')


# In[ ]:


rbf_results3.sort_values(by=['accuracy', 'precision', 'recall'], ascending=False).head(20)


#  ## **Sigmoid kernel**

# In[ ]:


param_grid_sigm = {'C': [0.001, 0.01, 0.1, 0.5, 0.8, 1, 1.5, 5, 10, 20, 50],
                   'class_weight': [None, 'balanced', {1:1.5, 0:1}, {1:2, 0:1}],
                   'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10, 100],
                   'coef0': [0, 0.1, 0.5, 1]}


# **Dane z lipca**

# In[ ]:


sigmoid_results = GridSearchCV_SVM(folds_07_2019_stand, folds_07_2019_labels, param_grid_sigm, kernel='sigmoid')


# In[ ]:


sigmoid_results.sort_values(by=['accuracy', 'precision', 'jaccard_score'], ascending=False)


#  **Dane z sierpnia**

# In[ ]:


sigmoid_results2 = GridSearchCV_SVM(folds_08_2019_stand, folds_08_2019_labels, param_grid_sigm, kernel='sigmoid')


# In[ ]:


sigmoid_results.sort_values(by=['accuracy', 'precision', 'jaccard_score'], ascending=False)


# ## **Wizualizacje wybranych modeli**

# * **Linear SVM dla danych z lipca 2019 (predykcje na danych z lipca 2021)**
# 
# * z parametrami: penalty: **l2** ,  C=**1** ,  loss="squared-hinge" ,  class_weight= **None** , intercept_scaling= 0.1
# 
# 

# In[ ]:


model = LinearSVC(penalty='l2', C=1, loss='squared_hinge', class_weight=None, intercept_scaling=1, dual=False)
df_train = July_2019_DF.drop(columns=['lon','lat', 'Year', 'Month', 'SoilT_40_100cm'])
nasa_07_2021 = pd.DataFrame(nasa2021full[nasa2021full['Month']==7])
nasa_fit = nasa_07_2021.drop(columns=['lon', 'lat', 'Year', 'Month', 'SoilT_40_100cm'])
y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia')
labels = pd.DataFrame({'lon': nasa_07_2021['lon'], 'lat': nasa_07_2021['lat'], 'label': y_pred})


# In[ ]:


colormap = get_colormap([0, 1], ['green', 'yellow'])
output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap), alpha=0.5)


# * **SVM z polynomial kernel dla danych z lipca 2019 (predykcje na danych z lipca 2021)**
# 
# * z parametrami: degree=**2** , C=**0.001** , class_weight={1:1.5, 0:1} , 	gamma =10 ,	coef0= **0.5**

# In[ ]:


model = SVC(kernel='poly', degree=2, C=0.001, class_weight={1:1.5, 0:1}, gamma=10, coef0=0.5)
y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia')
labels = pd.DataFrame({'lon': nasa_07_2021['lon'], 'lat': nasa_07_2021['lat'], 'label': y_pred})


# In[ ]:


output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap), alpha=0.5)


# * **SVM z jądrem RBF dla danych z lipca ((predykcje na danych z lipca 2021)**
# 
# * z parametrami: C=**0.1** ,  class_weight= {1:2, 0:1} , gamma=**scale**

# In[ ]:


model = SVC(kernel='rbf', C=0.1, class_weight={1:2, 0:1}, gamma='scale')
y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia')
labels = pd.DataFrame({'lon': nasa_07_2021['lon'], 'lat': nasa_07_2021['lat'], 'label': y_pred})


# In[ ]:


output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap), alpha=0.5)


# * **SVM z jądrem RBF dla danych z lipca ((predykcje na danych z lipca 2021)**
# 
# * z parametrami: C=**10** ,  class_weight= {1:1.5, 0:1} , gamma=**0.01**

# In[ ]:


model = SVC(kernel='rbf', C=10, class_weight={1:1.5, 0:1}, gamma=0.01)
y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia')
labels = pd.DataFrame({'lon': nasa_07_2021['lon'], 'lat': nasa_07_2021['lat'], 'label': y_pred})


# In[ ]:


output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap), alpha=0.5)


# * **SVM z sigmoid kernel dla danych z lipca (predykcje na danych z lipca 2021)**
# * z parametrami: C=**50** ,  class_weight=None , gamma=**0.01**, coef0=0

# In[ ]:


model = SVC(kernel='sigmoid',C=50, class_weight=None, gamma=0.01, coef0=0)
y_pred = train_and_predict(model, df_train, nasa_fit, 'pustynia')
labels = pd.DataFrame({'lon': nasa_07_2021['lon'], 'lat': nasa_07_2021['lat'], 'label': y_pred})


# In[ ]:


output_notebook()
show(plot_map(df=labels, parameter_name='label', colormap=colormap), alpha=0.5)

