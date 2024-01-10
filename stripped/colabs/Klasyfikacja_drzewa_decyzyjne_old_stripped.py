!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q dlcdn.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
!tar xf spark-3.5.0-bin-hadoop3.tgz
!pip install -q findspark

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.5.0-bin-hadoop3"

import findspark
findspark.init()

import pandas as pd
from google.colab import drive
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType, StringType, StructType

spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()

drive.mount('/content/drive')

columns = ['lon', 'lat', 'Date', 'Rainf', 'Evap', 'AvgSurfT', 'Albedo','SoilT_10_40cm', 'GVEG', 'PotEvap', 'RootMoist', 'SoilM_100_200cm']

# Utworzenie schematu określającego typ zmiennych
schema = StructType()
for i in columns:
  if i == "Date":
    schema = schema.add(i, IntegerType(), True)
  else:
    schema = schema.add(i, FloatType(), True)

# Wczytanie zbioru Nasa w sparku
nasa = spark.read.format('csv').option("header", True).schema(schema).load('/content/drive/MyDrive/BigMess/NASA/NASA.csv')
nasa.show(5)

nasa.createOrReplaceTempView("nasa")


nasa_ym = spark.sql("""
          SELECT
          CAST(SUBSTRING(CAST(Date AS STRING), 1, 4) AS INT) AS Year,
          CAST(SUBSTRING(CAST(Date AS STRING), 5, 2) AS INT) AS Month,
          n.*
          FROM nasa n
          """)
nasa_ym = nasa_ym.drop("Date")


nasa_ym.createOrReplaceTempView("nasa_ym")

from typing import List
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

NASA_sample_an = pd.read_csv('/content/drive/MyDrive/BigMess/NASA/NASA_an.csv',sep=';')

{1:NASA_sample_an['pustynia'].sum(), 2:NASA_sample_an['step'].sum() , 3:NASA_sample_an['pustynia'].count()- NASA_sample_an['pustynia'].sum()- NASA_sample_an['step'].sum()}

NASA_sample_an['klasa'] = np.where(
                        NASA_sample_an['pustynia'] == 1, 1, np.where(
                        NASA_sample_an['step'] == 1,2,3)
                        )

NASA_sample_an.rename(columns = {'lon':'lon_sample', 'lat':'lat_sample'}, inplace = True)

NASA_sample_an

spark_NASA_sample_an=spark.createDataFrame(NASA_sample_an)

spark_NASA_07_23 = spark.sql("""
                        SELECT
                        lon, lat, Rainf, Evap, AvgSurfT, Albedo, SoilT_10_40cm, GVEG, PotEvap, RootMoist, SoilM_100_200cm
                        FROM nasa_ym
                        WHERE Year = 2023 and Month = 7
                        """)

spark_NASA_sample_all = spark_NASA_07_23.join(
    spark_NASA_sample_an,
     [spark_NASA_07_23.lon==spark_NASA_sample_an.lon_sample , spark_NASA_07_23.lat==spark_NASA_sample_an.lat_sample],
    "inner"
    )

spark_NASA_sample_all=spark_NASA_sample_all.drop('lon_sample','lat_sample')

spark_NASA_sample_all.show(2)

print((spark_NASA_sample_all.count(), len(spark_NASA_sample_all.columns)))

pd_NASA_sample_all = spark_NASA_sample_all.toPandas()

pd_NASA_sample_all.head(5)

pd_NASA_sample_all.isnull().sum()

X = pd_NASA_sample_all.loc[:,'Rainf':'SoilM_100_200cm']
y_m1 = pd_NASA_sample_all['klasa']
y_m2 = pd_NASA_sample_all['pustynia']
y_m3 = pd_NASA_sample_all['step']

def information_gain(X: pd.DataFrame, y: pd.DataFrame) -> None:
  importances = mutual_info_classif(X, y)
  feature_info = pd.Series(importances, X.columns).sort_values(ascending=False)
  print(feature_info)

def correlations(X: pd.DataFrame) -> None:
  cor = X.corr()
  sns.heatmap(cor, annot=True,cmap='Reds')

def plot_data_dist(y: pd.DataFrame) -> None:
  dane = pd.Series(y).value_counts().sort_index()
  labels = list(np.sort(pd.unique(y)))
  ypos=np.arange(len(labels))
  plt.xticks(ypos, labels)
  plt.xlabel('Klasa')
  plt.ylabel('Czestosc')
  plt.title('Liczebnosc dla proby')
  plt.bar(ypos,dane)

def summary_model(model, X:pd.DataFrame, y:pd.DataFrame, labels_names: List) -> None:
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

def print_classification_report(model, X: pd.DataFrame, y: pd.DataFrame) -> None:
  y_predict = model.predict(X)
  print(classification_report(y, y_predict))

from typing import Optional, Tuple
from pandas import DataFrame
from imblearn.over_sampling import RandomOverSampler, SMOTE

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

X_m1_train, X_m1_test, y_m1_train, y_m1_test = train_test_split(X, y_m1, test_size=0.2, random_state=2023)

information_gain(X_m1_train, y_m1_train)

correlations(X_m1_train)

X_m1_train = X_m1_train.drop('SoilM_100_200cm',axis=1)
X_m1_test = X_m1_test.drop('SoilM_100_200cm',axis=1)
X_m1_train = X_m1_train.drop('PotEvap',axis=1)
X_m1_test = X_m1_test.drop('PotEvap',axis=1)

plot_data_dist(y_m1_train)

X_m1_train_bal, y_m1_train_bal = BalanceDataSet(X_m1_train, y_m1_train).useSMOTE()

plot_data_dist(y_m1_train_bal)

tree_classifier_m1 = tree.DecisionTreeClassifier(random_state = 2023)

tree_classifier_m1.fit(X_m1_train_bal, y_m1_train_bal)

print("classifier accuracy {:.2f}%".format(tree_classifier_m1.score(X_m1_test,  y_m1_test) * 100))

%matplotlib inline
plt.style.use("classic")
plt.figure(figsize=(30,20))
tree.plot_tree(tree_classifier_m1, max_depth=3, feature_names=X_m1_train.columns);

summary_model(tree_classifier_m1, X_m1_train_bal, y_m1_train_bal, ['1','2','3'])

print_classification_report(tree_classifier_m1, X_m1_train_bal, y_m1_train_bal)

summary_model(tree_classifier_m1, X_m1_test, y_m1_test, ['1','2','3'])

print_classification_report(tree_classifier_m1, X_m1_test, y_m1_test)

model_m1_path='/content/drive/MyDrive/BigMess/NASA/Modele/Klasyfikacja/tree_classifier_m1'

#with open(model_m1_path, 'wb') as files:
#    pickle.dump(tree_classifier_m1, files)

X_m2_train, X_m2_test, y_m2_train, y_m2_test = train_test_split(X, y_m2, test_size=0.2, random_state=2023)

information_gain(X_m2_train, y_m2_train)

correlations(X_m2_train)

X_m2_train = X_m2_train.drop('SoilM_100_200cm',axis=1)
X_m2_test = X_m2_test.drop('SoilM_100_200cm',axis=1)
X_m2_train = X_m2_train.drop('PotEvap',axis=1)
X_m2_test = X_m2_test.drop('PotEvap',axis=1)

plot_data_dist(y_m2_train)

X_m2_train_bal, y_m2_train_bal = BalanceDataSet(X_m2_train, y_m2_train).useSMOTE()

plot_data_dist(y_m2_train_bal)

tree_classifier_m2 = tree.DecisionTreeClassifier(random_state = 2023)

tree_classifier_m2.fit(X_m2_train_bal, y_m2_train_bal)

print("classifier accuracy {:.2f}%".format(tree_classifier_m2.score(X_m2_test,  y_m2_test) * 100))

%matplotlib inline
plt.style.use("classic")
plt.figure(figsize=(30,20))
tree.plot_tree(tree_classifier_m2, max_depth=3, feature_names=X_m2_train.columns);

summary_model(tree_classifier_m2, X_m2_train_bal, y_m2_train_bal, ['0', '1'])

print_classification_report(tree_classifier_m2, X_m2_train_bal, y_m2_train_bal)

summary_model(tree_classifier_m2, X_m2_test, y_m2_test, ['0', '1'])

print_classification_report(tree_classifier_m2, X_m2_test, y_m2_test)

model_m2_path='/content/drive/MyDrive/BigMess/NASA/Modele/Klasyfikacja/tree_classifier_m2'

#with open(model_m2_path, 'wb') as files:
#    pickle.dump(tree_classifier_m2, files)

X_m3_train, X_m3_test, y_m3_train, y_m3_test = train_test_split(X, y_m3, test_size=0.2, random_state=2023)

information_gain(X_m3_train, y_m3_train)

correlations(X_m3_train)

X_m3_train = X_m3_train.drop('SoilM_100_200cm',axis=1)
X_m3_test = X_m3_test.drop('SoilM_100_200cm',axis=1)
X_m3_train = X_m3_train.drop('SoilT_10_40cm',axis=1)
X_m3_test = X_m3_test.drop('SoilT_10_40cm',axis=1)
X_m3_train = X_m3_train.drop('AvgSurfT',axis=1)
X_m3_test = X_m3_test.drop('AvgSurfT',axis=1)
X_m3_train = X_m3_train.drop('PotEvap',axis=1)
X_m3_test = X_m3_test.drop('PotEvap',axis=1)

plot_data_dist(y_m3_train)

X_m3_train_bal, y_m3_train_bal = BalanceDataSet(X_m3_train, y_m3_train).useSMOTE()

plot_data_dist(y_m3_train_bal)

tree_classifier_m3 = tree.DecisionTreeClassifier(random_state = 2023)

tree_classifier_m3.fit(X_m3_train_bal, y_m3_train_bal)

print("classifier accuracy {:.2f}%".format(tree_classifier_m3.score(X_m3_test,  y_m3_test) * 100))

%matplotlib inline
plt.style.use("classic")
plt.figure(figsize=(30,20))
tree.plot_tree(tree_classifier_m3, max_depth=3, feature_names=X_m3_train.columns);

summary_model(tree_classifier_m3, X_m3_train_bal, y_m3_train_bal, ['0', '1'])

print_classification_report(tree_classifier_m3, X_m3_train_bal, y_m3_train_bal)

summary_model(tree_classifier_m3, X_m3_test, y_m3_test, ['0', '1'])

print_classification_report(tree_classifier_m3, X_m3_test, y_m3_test)

model_m3_path='/content/drive/MyDrive/BigMess/NASA/Modele/Klasyfikacja/tree_classifier_m3'

#with open(model_m3_path, 'wb') as files:
#    pickle.dump(tree_classifier_m3, files)

model_m1_path='/content/drive/MyDrive/BigMess/NASA/Modele/Klasyfikacja/tree_classifier_m1'
model_m2_path='/content/drive/MyDrive/BigMess/NASA/Modele/Klasyfikacja/tree_classifier_m2'
model_m3_path='/content/drive/MyDrive/BigMess/NASA/Modele/Klasyfikacja/tree_classifier_m3'

with open(model_m1_path , 'rb') as f:
    model = pickle.load(f)
