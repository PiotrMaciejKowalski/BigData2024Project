"""
<a href="https://colab.research.google.com/github/PiotrMaciejKowalski/BigData2024Project/blob/Analiza-szeregow-czasowych-dot-roslinnosci/colabs/Analiza_szeregow_czasowych_dot_roslinnosci.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# Wczytywanie danych w sparku
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


from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from google.colab import drive
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.types import IntegerType, FloatType, StringType, StructType
import IPython
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics import tsaplots
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings('ignore')


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

columns = ['lon', 'lat', 'Date', 'SWdown', 'LWdown', 'SWnet', 'LWnet', 'Qle', 'Qh', 'Qg', 'Qf', 'Snowf', 'Rainf', 'Evap', 'Qs', 'Qsb', 'Qsm', 'AvgSurfT', 'Albedo', 'SWE', 'SnowDepth', 'SnowFrac', 'SoilT_0_10cm', 'SoilT_10_40cm',
           'SoilT_40_100cm', 'SoilT_100_200cm', 'SoilM_0_10cm', 'SoilM_10_40cm', 'SoilM_40_100cm', 'SoilM_100_200cm', 'SoilM_0_100cm', 'SoilM_0_200cm', 'RootMoist', 'SMLiq_0_10cm', 'SMLiq_10_40cm', 'SMLiq_40_100cm', 'SMLiq_100_200cm',
           'SMAvail_0_100cm', 'SMAvail_0_200cm', 'PotEvap', 'ECanop', 'TVeg', 'ESoil', 'SubSnow', 'CanopInt', 'ACond', 'CCond', 'RCS', 'RCT', 'RCQ', 'RCSOL', 'RSmin','RSMacr', 'LAI', 'GVEG', 'Streamflow']

# Utworzenie schematu określającego typ zmiennych
schema = StructType()
for i in columns:
  if i == "Date":
    schema = schema.add(i, StringType(), True)
  else:
    schema = schema.add(i, FloatType(), True)


# Wczytanie zbioru Nasa w sparku
nasa = spark.read.format('csv').option("header", True).schema(schema).load('/content/drive/MyDrive/BigMess/NASA/NASA.csv')


"""
Zanim zaczniemy pisać kwerendy należy jeszcze dodać nasz DataFrame (df) do "przestrzeni nazw tabel" Sparka:
"""

nasa.createOrReplaceTempView("nasa")


"""
Rozdzielenie kolumny "Date" na kolumny "Year" oraz "Month"
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
# Analiza szergów czasowych - tylko wybrane kolumny z pliku "research.md"
"""

# Wyzanczenie unikatowych par współrzednych ze zbioru Nasa i zapisanie w Pandas
%%time
distinct_wsp = spark.sql("""
                          SELECT DISTINCT lon, lat FROM nasa
                          """).toPandas()


distinct_wsp.shape


def extract_time_series_from_dataframe(sdf: SparkDataFrame, lon: float, lat: float) -> pd.DataFrame:
  """
  Funkcja przekształcająca zadany SparkDataFrame w szereg czasowy dla zadanych współrzednych lon, lat.
  Parametry:
  - df (SparkDataFrame): ramka danych w Sparku zawierająca następujace kolumny: lon, lat, Date (pomiar ustawiony na pierwszy dzien miesiąca), atrybuty
  - lon (float): długość geograficzna
  - lat (float): szerokość geograficzna
  """
  # ograniczenie zbioru do konkretnej pary współrzędnych
  time_series = sdf.filter((sdf['lon'] == lon) & (sdf['lat'] == lat))
  # Przejście na pandas
  time_series_Pandas = time_series.toPandas()
  # Ustawienie 'date' jako indeksu
  time_series_Pandas.set_index('Date', inplace=True)
  return time_series_Pandas


def plot_time_series(df: pd.DataFrame, attribute: str) -> None:
  """
  Funkcja generująca wykres w czasie dla zadanego atrybutu.
  Parametry:
  - df (DataFrame): szereg czasowy; Pandas DataFrame
  - attribute (str): atrybut, dla którego chcemy zrobić wykres w czasie
  """
  assert attribute in df.columns, f"The attribute '{attribute}' is not a column in the DataFrame."
  # Obliczanie 12-miesięcznej średniej kroczącej
  df['12m_MA'] = df[attribute].rolling(window=12).mean()
  # Tworzenie wykresu
  plt.figure(figsize=(11,6))
  gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
  xmin, xmax = df.index.min() - pd.DateOffset(years=1) , df.index.max() + pd.DateOffset(years=1)

  ax1 = plt.subplot(gs[0])
  ax1.plot(df.index, df[attribute])
  ax1.plot(df.index, df['12m_MA'], label='12-miesięczna średnia krocząca', color='red')
  ax1.set_title(f"Wykres Szeregu Czasowego dla lon = {df['lon'].unique()}, lat = {df['lat'].unique()}")
  ax1.legend(loc='upper left')
  ax1.set_ylabel(f"Wartość {attribute}")
  ax1.set_xlim(xmin, xmax)
  ax1.grid(True)

  ax2 = plt.subplot(gs[1])
  ax2.plot(df.index, df['12m_MA'], label='12-miesięczna średnia krocząca', color='red')
  ax2.set_xlabel('Data')
  ax2.grid(True)
  ax2.set_xlim(xmin, xmax)

  plt.tight_layout()
  plt.show()


def perform_augumented_dickey_fuller_print_result(df: pd.DataFrame, attribute: str, prt: bool) -> str:
    """Funkcja do sprawdzania stacjonarności szeregu czasowego. Wykorzystuje ona Augmented Dickey-Fuller unit root test.
    Źródło: https://machinelearningmastery.com/time-series-data-stationary-python/
    Parametry:
    - df (Pandas DataFrame): ramka danych w Pandas zawierająca co najmniej następujace kolumny: atrybut,
    - attribute (str): nazwa kolumny, dla której będziemy sprawdzać stacjonarność,
    - prt (bool): zmienna determinująca, czy wypisać statystyke testu i wartości krytyczne.
    """

    result = adfuller(df[attribute].values, regression='ct')

    if prt:
      print(f'ADF Statistic: {round(result[0],4)}')
      print(f'p-value: {round(result[1], 4)}')
      print('Critical Values:')
      for key, value in result[4].items():
          print(f'{key}, {round(value,3)}')

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        return("Stationary")
    else:
        return("Non-stationary")


def plot_predictions(test_data: pd.DataFrame, test_pred: pd.DataFrame, attribute: str) -> None:
    """
    Funkcja tworzy wykres predykcji modelu i prawdziwych wartości.
    Parametry:
    - test_data (Pandas DataFrame): zbiór testowy, który zawiera badany atrybut,
    - test_pred (Pandas DataFrame): zbiór zawierający predykcje modelu,
    - attribute (str): nazwa kolumny będąca zmienną zależną (dependent variable).
    """
    plt.figure(figsize=(7, 4), dpi=100)
    test_data[attribute].plot(label=f'Prawdziwa wartość {attribute}', color='green')
    plt.plot(test_data[attribute].index, test_pred, label='Predykcja modelu', color='purple')
    plt.title(f'Wykres predykcji modelu dla {attribute}')
    plt.xlabel('Data')
    plt.ylabel(attribute)
    plt.legend(loc='upper left')


def train_test_split_by_year(df: pd.DataFrame, year_split: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """
  Funckja dzieli ramke danych na zbiór treningowy i testowy.
  - df (Pandas DataFrame): ramka danych w Pandas zawierająca co najmniej następujace kolumny: Year, Month, atrybut,
  - year_split (int): rok rozdzielający zbiór treningowy i testowy.
  """
  train_data =  df.loc[df['Year'] < year_split]
  test_data = df.loc[df['Year'] >= year_split]
  return train_data, test_data


def fit_linear_model_and_plot_predictions(df: pd.DataFrame, attribute: str, year_split: int) -> None:
  """
  Funkcja wykonuje dummy encoding na kolumnie miesiąc i dopasowuje model regresji liniowej dla jednej lokalizacji, a natępnie generuje wykres predykcji modelu.
  Parametry:
  - df (Pandas DataFrame): ramka danych w Pandas zawierająca co najmniej następujace kolumny: Year, Month, atrybut,
  - attribute (str): nazwa kolumny będąca zmienną zależną (dependent variable),
  - year_split (int): rok rozdzielający zbiór treningowy i testowy.
  """
  assert attribute in df.columns, f"The attribute '{attribute}' is not a column in the DataFrame."
  #Stworzenie nowej ramki danych, która będzie zawierać dummy variables.
  df_pd = df.copy(deep=True)
  # Dummy encoding
  df_pd = pd.get_dummies(df, columns = ['Month'])
  # Podział na zbiór treningowy i testowy
  train_data, test_data = train_test_split_by_year(df_pd, year_split)
  #Trenowanie modelu
  #Ponieważ funckja pd.get_dummies tworzy 12 nowych kolumn, na podstawie których będziemy trenować nasz model, wybieramy je w sposób, który nie popsuje się przy zmiennej ilości kolumn w zbiorze df.
  regression = LinearRegression().fit(train_data.iloc[:,-12:], train_data[attribute]) #FIXME zdecydowanie zła praktyka w kodzie. Naprawić przy pierwszej możliwość. NIE KOPIOWAĆ
  #Predykcja modelem
  df_pred = regression.predict(test_data.iloc[:,-12:])
  #Wypisanie metryk
  print(f"Mean squared error: {round(mean_squared_error(test_data[attribute], df_pred),3)}")
  print(f"Coefficient of determination: {round(r2_score(test_data[attribute], df_pred),3)}")
  #Stworzenie wykresu
  plot_predictions(test_data, df_pred, attribute)


def fit_sarima_and_plot_predictions(df: pd.DataFrame, attribute: str, year_split: int, p:int, d:int, q:int, P:int, D:int, Q:int, s:int) -> None:
  """
  Funkcja wykonuje dummy encoding na kolumnie miesiąc i dopasowuje model regresji liniowej dla jednej lokalizacji, a natępnie generuje wykres predykcji modelu.
  Parametry:
  - df (Pandas DataFrame): ramka danych w Pandas zawierająca co najmniej następujace kolumny: Date (pomiar ustawiony na pierwszy dzien miesiąca), atrybut,
  - attribute (str): nazwa kolumny będąca zmienną zależną (dependent variable),
  - year_split (int): rok rozdzielający zbiór treningowy i testowy,
  - p, d, q (int): parametry funkcji SARIMAX, argumentu order,
  - P, D, Q, s (int): parametry funkcji SARIMAX, argumantu seasonal_order.
  """
  assert attribute in df.columns, f"The attribute '{attribute}' is not a column in the DataFrame."
  # Stworzenie zbioru treningowego i testowego
  train_data, test_data = train_test_split_by_year(df, year_split)
  # Sprawdzenie stacjonarności przy użyciu funkcji perform_augumented_dickey_fuller_print_result
  if perform_augumented_dickey_fuller_print_result(df, attribute, False) == "Stationary":
    enforce_stat = False
  else:
    enforce_stat = True
  # Trenowanie modelu
  sarima = SARIMAX(train_data[attribute],
                order=(p,d,q),
                seasonal_order=(P,D,Q,s),
                trend="ct",
                enforce_stationarity=enforce_stat).fit()
  # Test modelu i wyświetlenie wyników
  df_pred = sarima.predict(start= str(year_split) + "-01-01", end="2023-09-01", steps=len(test_data))
  # Wyświetlenie metryk
  print(f"Mean squared error: {round(mean_squared_error(test_data[attribute], df_pred),3)}")
  print(f"Coefficient of determination: {round(r2_score(test_data[attribute], df_pred),3)}")
  # Tworzenie wykresu predykcji
  plot_predictions(test_data, df_pred, attribute)


def plot_variable_to_check_yearly_seasonality(df: pd.DataFrame, attribute: str) -> None:
  """
  Funkcja generująca wykres nakładający na siebie wartości zmiennej w kolejnych latach.
  Parametry:
  - df (Pandas DataFrame): ramka danych w Pandas zawierająca co najmniej następujace kolumny: Year, Month, atrybut,
  - attribute (str): nazwa kolumny, której wartości chcemy przedstawić na wykresie,
  """
  assert attribute in df.columns, f"The attribute '{attribute}' is not a column in the DataFrame."
  # Tworzenie wykresu
  df.pivot_table(index='Month',  columns='Year', values=attribute).plot()
  plt.ylabel(attribute)
  plt.title(f"Wskaźnik {attribute}, każda krzywa reprezentuje rok")
  plt.legend().remove()
  plt.show()


def plot_linear_and_polynomial_trend(df: pd.DataFrame, attribute: str, degree: int) -> None:
  """
  Funkcja generuje wykres zawierający trend liniowy i wielomianowy wybranego stopnia.
  Parametry:
  - df (Pandas DataFrame): ramka danych w Pandas zawierająca co najmniej następujace kolumny: Date (pomiar ustawiony na pierwszy dzien miesiąca), atrybut,
  - attribute (str): nazwa kolumny, dla której będziemy rysować linie trendu,
  - degree (int): stopień wielomianu.
  """
  assert attribute in df.columns, f"The attribute '{attribute}' is not a column in the DataFrame."
  # Dopasowanie prostej regresji do danych
  x = df.index.factorize()[0].reshape(-1, 1)+1
  regression = LinearRegression().fit(x, df[attribute])
  # Dopasowanie wielomianu do danych
  poly_y = np.polyfit(x[:,0], df[attribute].values, degree)
  # Pokazanie wzoru trendu
  print(f"Wzór trendu liniowego: {round(regression.coef_[0], 6)} * x + {round(regression.intercept_, 6)}")
  string = "".join([str(round(v, 6)) + " * x^" + str(degree - i) + " + " for i, v in enumerate(poly_y[:-2])])
  print(f"Wzór trendu wielomianowego: {string + str(round(poly_y[-2],6))} * x + {str(round(poly_y[-1],6))}")
  # Tworzenie wykresu
  plt.figure(figsize=(15, 5))
  plt.plot(df.index, df[attribute])
  plt.plot(df.index, regression.coef_ * x + regression.intercept_, color="blue", linewidth=3, label="Trend liniowy")
  plt.plot(df.index, np.polyval(poly_y, x[:,0]), color="red", linewidth=3, label="Trend wielomianowy")
  plt.title(f'Wykres trendu liniowego i wielomianowego o {degree} stopniach')
  plt.xlabel('Data')
  plt.ylabel(attribute)
  plt.legend(loc='upper left')
  plt.grid(True)
  plt.show()


def plot_acf_and_pacf(df: pd.DataFrame, attribute: str, lag: int) -> None:
  """
  Funkcja generująca wykres nakładający na siebie wartości zmiennej w kolejnych latach.
  Parametry:
  - df (Pandas DataFrame): ramka danych w Pandas zawierająca co najmniej następujace kolumny: Date (pomiar ustawiony na pierwszy dzien miesiąca), atrybut,
  - attribute (str): nazwa kolumny, której wartości chcemy przedstawić na wykresie,
  - lag (int): argument funkcji acf i pacf.
  """
  assert attribute in df.columns, f"The attribute '{attribute}' is not a column in the DataFrame."
  # Tworzenie wykresów
  fig, ax = plt.subplots(1,2,figsize=(15,5))
  tsaplots.plot_acf(df[attribute], lags=lag, ax=ax[0])
  ax[0].set_xlabel('Lag at k')
  ax[0].set_ylabel('Correlation coefficient')
  ax[0].set_title("Wykres autokorelacji")
  tsaplots.plot_pacf(df[attribute], lags=lag, ax=ax[1])
  ax[1].set_xlabel('Lag at k')
  ax[1].set_ylabel('Correlation coefficient')
  ax[1].set_title("Wykres autokorelacji cząstkowej")
  plt.show()


print("Lon Range:", distinct_wsp['lon'].min(), distinct_wsp['lon'].max())
print("Lat Range:", distinct_wsp['lat'].min(), distinct_wsp['lat'].max())


"""
Do przeprowadzenia analizy weźmy trzy lokalizacje:


*   obszar niepustynny,
*   obszar przejściowy,
*   obszar pustynny.

"""

# Obszar niepustynny
src = "https://www.globalforestwatch.org/map/country/USA/?map=eyJjZW50ZXIiOnsibGF0IjozOS45Mzc1LCJsbmciOi0xMjAuMTg3NDk5OTk5OTk5OTl9LCJ6b29tIjo5LjI1NDQyNDE0NjA1NDM4OCwiY2FuQm91bmQiOmZhbHNlLCJkYXRhc2V0cyI6W3siZGF0YXNldCI6InBvbGl0aWNhbC1ib3VuZGFyaWVzIiwibGF5ZXJzIjpbImRpc3B1dGVkLXBvbGl0aWNhbC1ib3VuZGFyaWVzIiwicG9saXRpY2FsLWJvdW5kYXJpZXMiXSwib3BhY2l0eSI6MSwidmlzaWJpbGl0eSI6dHJ1ZX0seyJkYXRhc2V0IjoidHJlZS1jb3Zlci1nYWluIiwibGF5ZXJzIjpbInRyZWUtY292ZXItZ2Fpbi0yMDAxLTIwMjAiXSwib3BhY2l0eSI6MSwidmlzaWJpbGl0eSI6dHJ1ZX0seyJkYXRhc2V0IjoidHJlZS1jb3Zlci1sb3NzIiwibGF5ZXJzIjpbInRyZWUtY292ZXItbG9zcyJdLCJvcGFjaXR5IjoxLCJ2aXNpYmlsaXR5Ijp0cnVlLCJ0aW1lbGluZVBhcmFtcyI6eyJzdGFydERhdGUiOiIyMDAxLTAxLTAxIiwiZW5kRGF0ZSI6IjIwMjItMTItMzEiLCJ0cmltRW5kRGF0ZSI6IjIwMjItMTItMzEifX0seyJkYXRhc2V0IjoidHJlZS1jb3ZlciIsImxheWVycyI6WyJ0cmVlLWNvdmVyLTIwMTAiXSwib3BhY2l0eSI6MSwidmlzaWJpbGl0eSI6dHJ1ZX1dfQ%3D%3D&mapMenu=eyJzZWFyY2hUeXBlIjoiZGVjaW1hbHMifQ%3D%3D&mapPrompts=eyJvcGVuIjp0cnVlLCJzdGVwc0tleSI6InJlY2VudEltYWdlcnkifQ%3D%3D"
IPython.display.IFrame(src, width=900, height=500)


# Wybranie punktu w danych, który jest blisko punktu (39.9268, -120.1052) i znajduje się w obszarze zaznaczonym na różowo na powyższej mapie.
distinct_wsp[(distinct_wsp['lon']<=-120) & (distinct_wsp['lon']>=-120.2) & (distinct_wsp['lat']<=40) & (distinct_wsp['lat']>=39.8)]


"""
Weźmy punkt nr 5888 ponieważ można na nim zaobserwować zmianę ilości drzew w ostatnich latach.
"""

# Obszar przejściowy
src = "https://www.globalforestwatch.org/map/country/USA/?map=eyJjZW50ZXIiOnsibGF0IjozOS44NDA1MDE5NTc5NDMzMSwibG5nIjotMTE5Ljc1ODA0NTUwOTkyMDczfSwiem9vbSI6OS4wMDMyMTc1NjA3Njc2MDIsImNhbkJvdW5kIjpmYWxzZSwiZGF0YXNldHMiOlt7ImRhdGFzZXQiOiJwb2xpdGljYWwtYm91bmRhcmllcyIsImxheWVycyI6WyJkaXNwdXRlZC1wb2xpdGljYWwtYm91bmRhcmllcyIsInBvbGl0aWNhbC1ib3VuZGFyaWVzIl0sIm9wYWNpdHkiOjEsInZpc2liaWxpdHkiOnRydWV9LHsiZGF0YXNldCI6InRyZWUtY292ZXItZ2FpbiIsImxheWVycyI6WyJ0cmVlLWNvdmVyLWdhaW4tMjAwMS0yMDIwIl0sIm9wYWNpdHkiOjEsInZpc2liaWxpdHkiOnRydWV9LHsiZGF0YXNldCI6InRyZWUtY292ZXItbG9zcyIsImxheWVycyI6WyJ0cmVlLWNvdmVyLWxvc3MiXSwib3BhY2l0eSI6MSwidmlzaWJpbGl0eSI6dHJ1ZSwidGltZWxpbmVQYXJhbXMiOnsic3RhcnREYXRlIjoiMjAwMS0wMS0wMSIsImVuZERhdGUiOiIyMDIyLTEyLTMxIiwidHJpbUVuZERhdGUiOiIyMDIyLTEyLTMxIn19LHsiZGF0YXNldCI6InRyZWUtY292ZXIiLCJsYXllcnMiOlsidHJlZS1jb3Zlci0yMDEwIl0sIm9wYWNpdHkiOjEsInZpc2liaWxpdHkiOnRydWV9XX0%3D&mapMenu=eyJzZWFyY2hUeXBlIjoiZGVjaW1hbHMiLCJzZWFyY2giOiJDZW50cmFsIEJhc2luIGFuZCBSYW5nZSJ9&mapPrompts=eyJvcGVuIjp0cnVlLCJzdGVwc0tleSI6InJlY2VudEltYWdlcnkifQ%3D%3D"
IPython.display.IFrame(src, width=900, height=500)


# Wybranie punktu w danych, który jest blisko punktu (39.8405, -119,75805).
distinct_wsp[(distinct_wsp['lon']<=-119.6) & (distinct_wsp['lon']>=-119.8) & (distinct_wsp['lat']<=40) & (distinct_wsp['lat']>=39.7)]


"""
Weźmy punkt nr 76158.
"""

# Obszar pustynny
src = "https://www.globalforestwatch.org/map/country/USA/?map=eyJjZW50ZXIiOnsibGF0IjozOS43NzcwNzMwNzg3MDQ1NTQsImxuZyI6LTExOC41NDA0NTc4NDkzOTQ5M30sInpvb20iOjkuNzUzNjE5NzU1ODYzMTEsImNhbkJvdW5kIjpmYWxzZSwiZGF0YXNldHMiOlt7ImRhdGFzZXQiOiJwb2xpdGljYWwtYm91bmRhcmllcyIsImxheWVycyI6WyJkaXNwdXRlZC1wb2xpdGljYWwtYm91bmRhcmllcyIsInBvbGl0aWNhbC1ib3VuZGFyaWVzIl0sIm9wYWNpdHkiOjEsInZpc2liaWxpdHkiOnRydWV9LHsiZGF0YXNldCI6InRyZWUtY292ZXItZ2FpbiIsImxheWVycyI6WyJ0cmVlLWNvdmVyLWdhaW4tMjAwMS0yMDIwIl0sIm9wYWNpdHkiOjEsInZpc2liaWxpdHkiOnRydWV9LHsiZGF0YXNldCI6InRyZWUtY292ZXItbG9zcyIsImxheWVycyI6WyJ0cmVlLWNvdmVyLWxvc3MiXSwib3BhY2l0eSI6MSwidmlzaWJpbGl0eSI6dHJ1ZSwidGltZWxpbmVQYXJhbXMiOnsic3RhcnREYXRlIjoiMjAwMS0wMS0wMSIsImVuZERhdGUiOiIyMDIyLTEyLTMxIiwidHJpbUVuZERhdGUiOiIyMDIyLTEyLTMxIn19LHsiZGF0YXNldCI6InRyZWUtY292ZXIiLCJsYXllcnMiOlsidHJlZS1jb3Zlci0yMDEwIl0sIm9wYWNpdHkiOjEsInZpc2liaWxpdHkiOnRydWV9XX0%3D&mapMenu=eyJzZWFyY2hUeXBlIjoiZGVjaW1hbHMiLCJzZWFyY2giOiJDZW50cmFsIEJhc2luIGFuZCBSYW5nZSJ9&mapPrompts=eyJvcGVuIjp0cnVlLCJzdGVwc0tleSI6InJlY2VudEltYWdlcnkifQ%3D%3D"
IPython.display.IFrame(src, width=900, height=500)


# Wybranie punktu w danych, który jest blisko punktu (39.77707, -118.54046).
distinct_wsp[(distinct_wsp['lon']<=-118.3) & (distinct_wsp['lon']>=-118.7) & (distinct_wsp['lat']<=39.9) & (distinct_wsp['lat']>=39.7)]


"""
Weźmy punkt nr 63828.
"""

"""
## **Rainf** (wskaźnik opadów deszczu)
"""

#Pomiaru zawsze na 1 dzień miesiąca
Rainf_SparkDataFrame = spark.sql("""
                        SELECT
                        lon, lat, Year, Month,
                        to_date(CONCAT(Year, '-', Month, '-1')) as Date, Rainf
                        FROM nasa_ym
                        order by lon, lat, Year, Month
                        """)


Rainf_time_series_nie_pustynia = extract_time_series_from_dataframe(Rainf_SparkDataFrame, -120.1875, 39.9375)
Rainf_time_series_przejsciowe = extract_time_series_from_dataframe(Rainf_SparkDataFrame, -119.6875, 39.8125)
Rainf_time_series_pustynia = extract_time_series_from_dataframe(Rainf_SparkDataFrame, -118.5625, 39.8125)
plot_time_series(Rainf_time_series_nie_pustynia, 'Rainf')
plot_time_series(Rainf_time_series_przejsciowe, 'Rainf')
plot_time_series(Rainf_time_series_pustynia, 'Rainf')


"""
Analiza dla obszaru niepustynnego.
"""

perform_augumented_dickey_fuller_print_result(Rainf_time_series_nie_pustynia, 'Rainf', True)


plot_acf_and_pacf(Rainf_time_series_nie_pustynia, "Rainf", 30)
seasonal_decompose(Rainf_time_series_nie_pustynia['Rainf'], model="additive", period=12).plot().show()


plot_variable_to_check_yearly_seasonality(Rainf_time_series_nie_pustynia, 'Rainf')
fit_linear_model_and_plot_predictions(Rainf_time_series_nie_pustynia, 'Rainf', 2020)


plot_linear_and_polynomial_trend(Rainf_time_series_nie_pustynia, 'Rainf', 5)


"""
Analiza dla obszaru przejściowego.
"""

perform_augumented_dickey_fuller_print_result(Rainf_time_series_przejsciowe, 'Rainf', True)


plot_acf_and_pacf(Rainf_time_series_przejsciowe, "Rainf", 30)
seasonal_decompose(Rainf_time_series_przejsciowe['Rainf'], model="additive", period=12).plot().show()


plot_variable_to_check_yearly_seasonality(Rainf_time_series_przejsciowe, 'Rainf')
fit_linear_model_and_plot_predictions(Rainf_time_series_przejsciowe, 'Rainf', 2020)


plot_linear_and_polynomial_trend(Rainf_time_series_przejsciowe, 'Rainf', 5)


"""
Analiza dla obszaru pustynnego.
"""

perform_augumented_dickey_fuller_print_result(Rainf_time_series_pustynia, 'Rainf', True)


plot_acf_and_pacf(Rainf_time_series_pustynia, "Rainf", 30)
seasonal_decompose(Rainf_time_series_pustynia['Rainf'], model="additive", period=12).plot().show()


plot_variable_to_check_yearly_seasonality(Rainf_time_series_pustynia, 'Rainf')
fit_linear_model_and_plot_predictions(Rainf_time_series_pustynia, 'Rainf', 2020)


plot_linear_and_polynomial_trend(Rainf_time_series_pustynia, 'Rainf', 5)


"""
## **Evap** (wskaźnik całkowitej ewapotranspiracji)
"""

#Pomiaru zawsze na 1 dzień miesiąca
Evap_SparkDataFrame = spark.sql("""
                        SELECT
                        lon, lat, Year, Month,
                        to_date(CONCAT(Year, '-', Month, '-1')) as Date, Evap
                        FROM nasa_ym
                        order by lon, lat, Year, Month
                        """)


Evap_time_series_nie_pustynia = extract_time_series_from_dataframe(Evap_SparkDataFrame, -120.1875, 39.9375)
Evap_time_series_przejsciowe = extract_time_series_from_dataframe(Evap_SparkDataFrame, -119.6875, 39.8125)
Evap_time_series_pustynia = extract_time_series_from_dataframe(Evap_SparkDataFrame, -118.5625, 39.8125)
plot_time_series(Evap_time_series_nie_pustynia, 'Evap')
plot_time_series(Evap_time_series_przejsciowe, 'Evap')
plot_time_series(Evap_time_series_pustynia, 'Evap')


"""
Analiza dla obszaru niepustynnego.
"""

perform_augumented_dickey_fuller_print_result(Evap_time_series_nie_pustynia, 'Evap', True)


plot_acf_and_pacf(Evap_time_series_nie_pustynia, "Evap", 30)
seasonal_decompose(Evap_time_series_nie_pustynia['Evap'], model="additive", period=12).plot().show()


plot_variable_to_check_yearly_seasonality(Evap_time_series_nie_pustynia, 'Evap')
fit_linear_model_and_plot_predictions(Evap_time_series_nie_pustynia, 'Evap', 2020)


plot_linear_and_polynomial_trend(Evap_time_series_nie_pustynia, 'Evap', 5)


"""
Analiza dla obszaru przejściowego.
"""

perform_augumented_dickey_fuller_print_result(Evap_time_series_przejsciowe, 'Evap', True)


plot_acf_and_pacf(Evap_time_series_przejsciowe, "Evap", 30)
seasonal_decompose(Evap_time_series_przejsciowe['Evap'], model="additive", period=12).plot().show()


plot_variable_to_check_yearly_seasonality(Evap_time_series_przejsciowe, 'Evap')
fit_linear_model_and_plot_predictions(Evap_time_series_przejsciowe, 'Evap', 2020)


plot_linear_and_polynomial_trend(Evap_time_series_przejsciowe, 'Evap', 5)


"""
Analiza dla obszaru pustynnego.
"""

perform_augumented_dickey_fuller_print_result(Evap_time_series_pustynia, 'Evap', True)


plot_acf_and_pacf(Evap_time_series_pustynia, "Evap", 30)
seasonal_decompose(Evap_time_series_pustynia['Evap'], model="additive", period=12).plot().show()


plot_variable_to_check_yearly_seasonality(Evap_time_series_pustynia, 'Evap')
fit_linear_model_and_plot_predictions(Evap_time_series_pustynia, 'Evap', 2020)


fit_sarima_and_plot_predictions(Evap_time_series_pustynia, 'Evap', 2020, 6,0,2, 3,1,0,12)


plot_linear_and_polynomial_trend(Evap_time_series_pustynia, 'Evap', 5)


"""
## **AvgSurfT** (wskaźnik średniej temperatury powierzchni ziemi)
"""

#Pomiaru zawsze na 1 dzień miesiąca
AvgSurfT_SparkDataFrame = spark.sql("""
                        SELECT
                        lon, lat, Year, Month,
                        to_date(CONCAT(Year, '-', Month, '-1')) as Date, AvgSurfT
                        FROM nasa_ym
                        order by lon, lat, Year, Month
                        """)


AvgSurfT_time_series_nie_pustynia = extract_time_series_from_dataframe(AvgSurfT_SparkDataFrame, -120.1875, 39.9375)
AvgSurfT_time_series_przejsciowe = extract_time_series_from_dataframe(AvgSurfT_SparkDataFrame, -119.6875, 39.8125)
AvgSurfT_time_series_pustynia = extract_time_series_from_dataframe(AvgSurfT_SparkDataFrame, -118.5625, 39.8125)
plot_time_series(AvgSurfT_time_series_nie_pustynia, 'AvgSurfT')
plot_time_series(AvgSurfT_time_series_przejsciowe, 'AvgSurfT')
plot_time_series(AvgSurfT_time_series_pustynia, 'AvgSurfT')


"""
Analiza dla obszaru niepustynnego.
"""

perform_augumented_dickey_fuller_print_result(AvgSurfT_time_series_nie_pustynia, 'AvgSurfT', True)


plot_acf_and_pacf(AvgSurfT_time_series_nie_pustynia, "AvgSurfT", 30)
seasonal_decompose(AvgSurfT_time_series_nie_pustynia['AvgSurfT'], model="additive", period=12).plot().show()


plot_variable_to_check_yearly_seasonality(AvgSurfT_time_series_nie_pustynia, 'AvgSurfT')
fit_linear_model_and_plot_predictions(AvgSurfT_time_series_nie_pustynia, 'AvgSurfT', 2020)


plot_linear_and_polynomial_trend(AvgSurfT_time_series_nie_pustynia, 'AvgSurfT', 5)


"""
Analiza dla obszaru przejściowego.
"""

perform_augumented_dickey_fuller_print_result(AvgSurfT_time_series_przejsciowe, 'AvgSurfT', True)


plot_acf_and_pacf(AvgSurfT_time_series_przejsciowe, "AvgSurfT", 30)
seasonal_decompose(AvgSurfT_time_series_przejsciowe['AvgSurfT'], model="additive", period=12).plot().show()


plot_variable_to_check_yearly_seasonality(AvgSurfT_time_series_przejsciowe, 'AvgSurfT')
fit_linear_model_and_plot_predictions(AvgSurfT_time_series_przejsciowe, 'AvgSurfT', 2020)


plot_linear_and_polynomial_trend(AvgSurfT_time_series_przejsciowe, 'AvgSurfT', 5)


"""
Analiza dla obszaru pustynnego.
"""

perform_augumented_dickey_fuller_print_result(AvgSurfT_time_series_pustynia, 'AvgSurfT', True)


plot_acf_and_pacf(AvgSurfT_time_series_pustynia, "AvgSurfT", 30)
seasonal_decompose(AvgSurfT_time_series_pustynia['AvgSurfT'], model="additive", period=12).plot().show()


plot_variable_to_check_yearly_seasonality(AvgSurfT_time_series_pustynia, 'AvgSurfT')
fit_linear_model_and_plot_predictions(AvgSurfT_time_series_pustynia, 'AvgSurfT', 2020)


plot_linear_and_polynomial_trend(AvgSurfT_time_series_pustynia, 'AvgSurfT', 5)


"""
## **Albedo** (wskaźnik albedo)
"""

#Pomiaru zawsze na 1 dzień miesiąca
Albedo_SparkDataFrame = spark.sql("""
                        SELECT
                        lon, lat, Year, Month,
                        to_date(CONCAT(Year, '-', Month, '-1')) as Date, Albedo
                        FROM nasa_ym
                        order by lon, lat, Year, Month
                        """)


Albedo_time_series_nie_pustynia = extract_time_series_from_dataframe(Albedo_SparkDataFrame, -120.1875, 39.9375)
Albedo_time_series_przejsciowe = extract_time_series_from_dataframe(Albedo_SparkDataFrame, -119.6875, 39.8125)
Albedo_time_series_pustynia = extract_time_series_from_dataframe(Albedo_SparkDataFrame, -118.5625, 39.8125)
plot_time_series(Albedo_time_series_nie_pustynia, 'Albedo')
plot_time_series(Albedo_time_series_przejsciowe, 'Albedo')
plot_time_series(Albedo_time_series_pustynia, 'Albedo')


"""
Analiza dla obszaru niepustynnego.
"""

perform_augumented_dickey_fuller_print_result(Albedo_time_series_nie_pustynia, 'Albedo', True)


plot_acf_and_pacf(Albedo_time_series_nie_pustynia, "Albedo", 30)
seasonal_decompose(Albedo_time_series_nie_pustynia['Albedo'], model="additive", period=12).plot().show()


plot_variable_to_check_yearly_seasonality(Albedo_time_series_nie_pustynia, 'Albedo')
fit_linear_model_and_plot_predictions(Albedo_time_series_nie_pustynia, 'Albedo', 2020)


fit_sarima_and_plot_predictions(Albedo_time_series_nie_pustynia, 'Albedo', 2020, 2,0,0, 3,1,2,12)


plot_linear_and_polynomial_trend(Albedo_time_series_nie_pustynia, 'Albedo', 5)


"""
Analiza dla obszaru przejściowego.
"""

perform_augumented_dickey_fuller_print_result(Albedo_time_series_przejsciowe, 'Albedo', True)


plot_acf_and_pacf(Albedo_time_series_przejsciowe, "Albedo", 30)
seasonal_decompose(Albedo_time_series_przejsciowe['Albedo'], model="additive", period=12).plot().show()


plot_variable_to_check_yearly_seasonality(Albedo_time_series_przejsciowe, 'Albedo')
fit_linear_model_and_plot_predictions(Albedo_time_series_przejsciowe, 'Albedo', 2020)


plot_linear_and_polynomial_trend(Albedo_time_series_przejsciowe, 'Albedo', 5)


"""
Analiza dla obszaru pustynnego.
"""

perform_augumented_dickey_fuller_print_result(Albedo_time_series_pustynia, 'Albedo', True)


plot_acf_and_pacf(Albedo_time_series_pustynia, "Albedo", 30)
seasonal_decompose(Albedo_time_series_pustynia['Albedo'], model="additive", period=12).plot().show()


plot_variable_to_check_yearly_seasonality(Albedo_time_series_pustynia, 'Albedo')
fit_linear_model_and_plot_predictions(Albedo_time_series_pustynia, 'Albedo', 2020)


fit_sarima_and_plot_predictions(Albedo_time_series_pustynia, 'Albedo', 2020, 1,0,0, 1,1,0,12)


plot_linear_and_polynomial_trend(Albedo_time_series_pustynia, 'Albedo', 5)


"""
## **SoilM_40_100cm** (wskaźnik zawartości wody w warstwie o głębokości od 40 do 100 cm)
"""

#Pomiaru zawsze na 1 dzień miesiąca.
SoilM_40_100cm_SparkDataFrame = spark.sql("""
                        SELECT
                        lon, lat, Year, Month,
                        to_date(CONCAT(Year, '-', Month, '-1')) as Date, SoilM_40_100cm
                        FROM nasa_ym
                        order by lon, lat, Year, Month
                        """)


SoilM_40_100cm_time_series_nie_pustynia = extract_time_series_from_dataframe(SoilM_40_100cm_SparkDataFrame, -120.1875, 39.9375)
SoilM_40_100cm_time_series_przejsciowe = extract_time_series_from_dataframe(SoilM_40_100cm_SparkDataFrame, -119.6875, 39.8125)
SoilM_40_100cm_time_series_pustynia = extract_time_series_from_dataframe(SoilM_40_100cm_SparkDataFrame, -118.5625, 39.8125)
plot_time_series(SoilM_40_100cm_time_series_nie_pustynia, 'SoilM_40_100cm')
plot_time_series(SoilM_40_100cm_time_series_przejsciowe, 'SoilM_40_100cm')
plot_time_series(SoilM_40_100cm_time_series_pustynia, 'SoilM_40_100cm')


"""
Analiza dla obszaru niepustynnego.
"""

perform_augumented_dickey_fuller_print_result(SoilM_40_100cm_time_series_nie_pustynia, 'SoilM_40_100cm', True)


plot_acf_and_pacf(SoilM_40_100cm_time_series_nie_pustynia, "SoilM_40_100cm", 30)
seasonal_decompose(SoilM_40_100cm_time_series_nie_pustynia['SoilM_40_100cm'], model="additive", period=12).plot().show()


plot_variable_to_check_yearly_seasonality(SoilM_40_100cm_time_series_nie_pustynia, 'SoilM_40_100cm')
fit_linear_model_and_plot_predictions(SoilM_40_100cm_time_series_nie_pustynia, 'SoilM_40_100cm', 2020)


fit_sarima_and_plot_predictions(SoilM_40_100cm_time_series_nie_pustynia, 'SoilM_40_100cm', 2020, 2,0,0, 1,1,0,12)


plot_linear_and_polynomial_trend(SoilM_40_100cm_time_series_nie_pustynia, 'SoilM_40_100cm', 5)


"""
Analiza dla obszaru przejściowego.
"""

perform_augumented_dickey_fuller_print_result(SoilM_40_100cm_time_series_przejsciowe, 'SoilM_40_100cm', True)


plot_acf_and_pacf(SoilM_40_100cm_time_series_przejsciowe, "SoilM_40_100cm", 30)
seasonal_decompose(SoilM_40_100cm_time_series_przejsciowe['SoilM_40_100cm'], model="additive", period=12).plot().show()


plot_variable_to_check_yearly_seasonality(SoilM_40_100cm_time_series_przejsciowe, 'SoilM_40_100cm')
fit_linear_model_and_plot_predictions(SoilM_40_100cm_time_series_przejsciowe, 'SoilM_40_100cm', 2020)


fit_sarima_and_plot_predictions(SoilM_40_100cm_time_series_przejsciowe, 'SoilM_40_100cm', 2020, 2,0,0, 1,1,0,12)


plot_linear_and_polynomial_trend(SoilM_40_100cm_time_series_przejsciowe, 'SoilM_40_100cm', 5)


"""
Analiza dla obszaru pustynnego.
"""

perform_augumented_dickey_fuller_print_result(SoilM_40_100cm_time_series_pustynia, 'SoilM_40_100cm', True)


plot_acf_and_pacf(SoilM_40_100cm_time_series_pustynia, "SoilM_40_100cm", 30)
seasonal_decompose(SoilM_40_100cm_time_series_pustynia['SoilM_40_100cm'], model="additive", period=12).plot().show()


plot_variable_to_check_yearly_seasonality(SoilM_40_100cm_time_series_pustynia, 'SoilM_40_100cm')
fit_linear_model_and_plot_predictions(SoilM_40_100cm_time_series_pustynia, 'SoilM_40_100cm', 2020)


fit_sarima_and_plot_predictions(SoilM_40_100cm_time_series_pustynia, 'SoilM_40_100cm', 2020, 1,0,1, 0,1,1,12)


plot_linear_and_polynomial_trend(SoilM_40_100cm_time_series_pustynia, 'SoilM_40_100cm', 5)


"""
## **GVEG** (wskaźnik roślinności)
"""

#Pomiaru zawsze na 1 dzień miesiąca.
GVEG_SparkDataFrame = spark.sql("""
                        SELECT
                        lon, lat, Year, Month,
                        to_date(CONCAT(Year, '-', Month, '-1')) as Date, GVEG
                        FROM nasa_ym
                        order by lon, lat, Year, Month
                        """)


GVEG_time_series_nie_pustynia = extract_time_series_from_dataframe(GVEG_SparkDataFrame, -120.1875, 39.9375)
GVEG_time_series_przejsciowe = extract_time_series_from_dataframe(GVEG_SparkDataFrame, -119.6875, 39.8125)
GVEG_time_series_pustynia = extract_time_series_from_dataframe(GVEG_SparkDataFrame, -118.5625, 39.8125)
plot_time_series(GVEG_time_series_nie_pustynia, 'GVEG')
plot_time_series(GVEG_time_series_przejsciowe, 'GVEG')
plot_time_series(GVEG_time_series_pustynia, 'GVEG')


"""
Analiza dla obszaru niepustynnego.
"""

perform_augumented_dickey_fuller_print_result(GVEG_time_series_nie_pustynia, 'GVEG', True)


plot_acf_and_pacf(GVEG_time_series_nie_pustynia, "GVEG", 30)
seasonal_decompose(GVEG_time_series_nie_pustynia['GVEG'], model="additive", period=12).plot().show()


plot_variable_to_check_yearly_seasonality(GVEG_time_series_nie_pustynia, 'GVEG')
fit_linear_model_and_plot_predictions(GVEG_time_series_nie_pustynia, 'GVEG', 2020)


plot_linear_and_polynomial_trend(GVEG_time_series_nie_pustynia, 'GVEG', 5)


"""
Analiza dla obszaru przejściowego.
"""

perform_augumented_dickey_fuller_print_result(GVEG_time_series_przejsciowe, 'GVEG', True)


plot_acf_and_pacf(GVEG_time_series_przejsciowe, "GVEG", 30)
seasonal_decompose(GVEG_time_series_przejsciowe['GVEG'], model="additive", period=12).plot().show()


plot_variable_to_check_yearly_seasonality(GVEG_time_series_przejsciowe, 'GVEG')
fit_linear_model_and_plot_predictions(GVEG_time_series_przejsciowe, 'GVEG', 2020)


plot_linear_and_polynomial_trend(GVEG_time_series_przejsciowe, 'GVEG', 5)


