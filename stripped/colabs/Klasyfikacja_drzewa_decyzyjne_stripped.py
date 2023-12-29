"""
# Klasyfikacja z wykorzystaniem drzew decyzyjnych
"""

"""
## Wczytywanie danych w sparku
"""

"""
Utworzenie środowiska pyspark do obliczeń:
"""

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


"""
Utowrzenie sesji:
"""

spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()


"""
Połączenie z dyskiem:
"""

drive.mount('/content/drive')


"""
Wczytanie danych NASA znajdujących się na dysku w sparku:
"""

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


"""
Rodzielenie kolumny "Date" na kolumny "Year" oraz "Month"
"""

nasa_ym = spark.sql("""
          SELECT
          CAST(SUBSTRING(CAST(Date AS STRING), 1, 4) AS INT) AS Year,
          CAST(SUBSTRING(CAST(Date AS STRING), 5, 2) AS INT) AS Month,
          n.*
          FROM nasa n
          """)
nasa_ym = nasa_ym.drop("Date")


nasa_ym.createOrReplaceTempView("nasa_ym")


"""
## Budowa modelu
"""

"""
**Cel**:

Celem jest zbudowanie modelu do klasyfikacji czy wskazany punkt lokalizacyjny ze zbioru danych NASA jest: pustynia, stepem lub innym obszarem.

**Proba danych**:

Dane wykorzystane do modelowania zostały stworzone po przez połączenie dwoch zbiorów danych:

* 100 lokalizacji *lon* i *lat* z określoną flagą 0, 1 w kolumnach *pustynia* lub *step* (reczna adnotacja)
* danych NASA od *pazdziernika 2022* do  *wrzesnia 2023*

**Metoda**:

Do modelowania uzyto metody drzew decyzyjnych.
"""

"""
### Import bibliotek
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


"""
### Przygotowanie danych
"""

NASA_sample_an = pd.read_csv('/content/drive/MyDrive/BigMess/NASA/NASA_sample_an.csv',sep=';')


"""
Dodanie kolumny *klasa* z nastepujacym oslownikowaniem:

* **1** - pustynia
* **2** - step
* **3** - inne
"""

NASA_sample_an['klasa'] = np.where(
                        NASA_sample_an['pustynia'] == 1, 1, np.where(
                        NASA_sample_an['step'] == 1,2,3)
                        )


NASA_sample_an=NASA_sample_an.drop(['pustynia', 'step'],axis=1)


NASA_sample_an.rename(columns = {'lon':'lon_sample', 'lat':'lat_sample'}, inplace = True)


NASA_sample_an


spark_NASA_sample_an=spark.createDataFrame(NASA_sample_an)


spark_NASA_22_23 = spark.sql("""
                        SELECT
                        lon, lat, Rainf, Evap, AvgSurfT, Albedo, SoilT_10_40cm, GVEG, PotEvap, RootMoist, SoilM_100_200cm
                        FROM nasa_ym
                        WHERE (Year = 2023 and Month in (1,2,3,4,5,6,7,8,9))
                        or (Year = 2022 and Month in (10,11,12))
                        """)


spark_NASA_sample_all = spark_NASA_22_23.join(
    spark_NASA_sample_an,
     [spark_NASA_22_23.lon==spark_NASA_sample_an.lon_sample , spark_NASA_22_23.lat==spark_NASA_sample_an.lat_sample],
    "inner"
    )


spark_NASA_sample_all=spark_NASA_sample_all.drop('lon_sample','lat_sample')


spark_NASA_sample_all.show(2)


print((spark_NASA_sample_all.count(), len(spark_NASA_sample_all.columns)))


pd_NASA_sample_all = spark_NASA_sample_all.toPandas()


pd_NASA_sample_all.head(5)


"""
Sprawdzenie, czy w danych występują braki - nie.
"""

pd_NASA_sample_all.isnull().sum()


X = pd_NASA_sample_all.loc[:,'Rainf':'SoilM_100_200cm']
y = pd_NASA_sample_all['klasa']


"""
### Analiza danych do modelowania
"""

"""
Zmienne kandydatki:

* **GVEG** - wskaznik roslinnosci
* **Rainf** - wskaznik opadow deszczu
* **Evap** - wskaznik calkowitej ewapotranspiracji
* **AvgSurfT** - wskaznik sredniej temperatury powierzchni ziemi
* **Albedo** - wskaznik albedo
* **SoilT_40_100cm** - wskaznik temperatury gleby w warstwie o glebokosci od 40 do 100 cm
* **PotEvap** - wskaznik potencjalnej ewapotranspiracji
* **RootMoist** - wilgotnosć gleby w strefie korzeniowej (parowanie, ktore mialoby miejsce, gdyby dostepne bylo wystarczajace zrodlo wody)
* **SoilM_100_200cm** - wilgotnosc gleby w warstwie o glebokosci od 100 do 200 cm


"""

"""
#### Podzial na zbior treningowy i testowy
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)


"""
#### Analiza jednoczynnikowa
"""

"""
Obliczmy zysk informacji.
"""

from sklearn.feature_selection import mutual_info_classif


importances = mutual_info_classif(X_train, y_train)
feature_info = pd.Series(importances, X_train.columns).sort_values(ascending=False)
feature_info


"""
#### Analiza wieloczynnikowa
"""

"""
Obliczmy korelacje zmiennych.
"""

cor = X.corr()
sns.heatmap(cor, annot=True,cmap='Reds')


"""
Usuniecie zmiennej *SoilT_10_40cm*. Cechuje sie ona malym zyskiem informacji i duza korelacja ze zmiennąa *AvgSurfT*.
"""

# na ten moment odstapilem od usuniecia, po usunieciu accuracy wychodzi 80% na testowym.
#X_train = X_train.drop('SoilT_10_40cm',axis=1)
#X_test = X_test.drop('SoilT_10_40cm',axis=1)


"""
#### Zbalansowanie datasetu
"""

def plot_data_dist(y: pd.DataFrame) -> None:
  dane = pd.Series(y).value_counts().sort_index()
  labels = list(np.sort(pd.unique(y)))
  ypos=np.arange(len(labels))
  plt.xticks(ypos, labels)
  plt.xlabel('Klasa')
  plt.ylabel('Czestosc')
  plt.title('Liczebnosc dla proby')
  plt.bar(ypos,dane)


plot_data_dist(y_train)


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


X_train_bal, y_train_bal = BalanceDataSet(X_train, y_train).useSMOTE()


plot_data_dist(y_train_bal)


"""
### Drzewa decyzyjne
"""

from sklearn import tree


tree_classifier = tree.DecisionTreeClassifier(random_state = 2023)


tree_classifier.fit(X_train_bal, y_train_bal)


print("classifier accuracy {:.2f}%".format(tree_classifier.score(X_test,  y_test) * 100))


%matplotlib inline
plt.style.use("classic")
plt.figure(figsize=(30,20))
tree.plot_tree(tree_classifier, max_depth=5, feature_names=X.columns);


"""
### Ocena modelu
"""

def summary_model(model, X:pd.DataFrame, y:pd.DataFrame) -> None:
  y_pred = model.predict(X)
  y_real= y
  cf_matrix = confusion_matrix(y_real, y_pred)
  group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
  group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
  labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
  labels = np.asarray(labels).reshape(3,3)
  sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Reds',xticklabels=['1','2','3'],yticklabels=['1','2','3'])
  plt.xlabel('Predykcja')
  plt.ylabel('Rzeczywistość')
  plt.show()


"""
Na danych testowych.
"""

summary_model(tree_classifier, X_test, y_test)


y_test_predict = tree_classifier.predict(X_test)


 print(classification_report(y_test,y_test_predict ))


"""
Na zbalansowanych danych testowych.
"""

X_test_bal, y_test_bal = BalanceDataSet(X_test, y_test).useSMOTE()


summary_model(tree_classifier, X_test_bal, y_test_bal)


"""
### Zapisanie modelu
"""

model_path='/content/drive/MyDrive/Colab Notebooks/Analiza BIG DATA/Sprint 2/Modele/tree_classifier'


with open(model_path, 'wb') as files:
    pickle.dump(tree_classifier, files)


"""
### Odczyt modelu
"""

with open(model_path , 'rb') as f:
    model = pickle.load(f)




