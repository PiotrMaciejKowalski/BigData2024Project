from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/PiotrMaciejKowalski/BigData2024Project.git
%cd BigData2024Project
!git checkout refactoring-sprint2
%cd ..

!chmod 755 /content/BigData2024Project/src/setup.sh
!/content/BigData2024Project/src/setup.sh

import sys
sys.path.append('/content/BigData2024Project/src')

from start_spark import initialize_spark
initialize_spark()

from pyspark.sql import SparkSession

from big_mess.loaders import preprocessed_loader

spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()

data=preprocessed_loader(spark,'/content/drive/MyDrive/BigMess/NASA/NASA_anotated_preprocessed.csv')

data.show()

from typing import List
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

df = (
    data
    .drop('SoilT_40_100cm') #remove after fixing null values there
    .toPandas()
)
df.head()

df['klasa'] = np.where(
                        df['pustynia'] == 1, 1, np.where(
                        df['step'] == 1,2,3)
                        )

{1:df['pustynia'].sum(), 2:df['step'].sum() , 3:df['pustynia'].count()- df['pustynia'].sum()- df['step'].sum()}

print(df.shape)

df.head(5)

df.isnull().sum()

X = df.loc[:,'Rainf':'SoilM_100_200cm']
y_m1 = df['klasa']
y_m2 = df['pustynia']
y_m3 = df['step']

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

X_m1_train = X_m1_train.drop('SoilT_10_40cm',axis=1)
X_m1_test = X_m1_test.drop('SoilT_10_40cm',axis=1)

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

X_m2_train = X_m2_train.drop('SoilT_10_40cm',axis=1)
X_m2_test = X_m2_test.drop('SoilT_10_40cm',axis=1)

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

X_m3_train = X_m3_train.drop('AvgSurfT',axis=1)
X_m3_test = X_m3_test.drop('AvgSurfT',axis=1)
X_m3_train = X_m3_train.drop('Evap',axis=1)
X_m3_test = X_m3_test.drop('Evap',axis=1)
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
