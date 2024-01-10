import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from prettytable import PrettyTable
from google.colab import drive
from google.colab import files

!pip install pyspark
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType, StringType, StructType
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.appName('Corr_Analysis').getOrCreate()

drive.mount('/content/drive')

columns = ['lon', 'lat', 'Date', 'SWdown', 'LWdown', 'SWnet', 'LWnet', 'Qle', 'Qh', 'Qg', 'Qf', 'Snowf', 'Rainf', 'Evap', 'Qs', 'Qsb', 'Qsm', 'AvgSurfT', 'Albedo', 'SWE', 'SnowDepth', 'SnowFrac', 'SoilT_0_10cm', 'SoilT_10_40cm',
           'SoilT_40_100cm', 'SoilT_100_200cm', 'SoilM_0_10cm', 'SoilM_10_40cm', 'SoilM_40_100cm', 'SoilM_100_200cm', 'SoilM_0_100cm', 'SoilM_0_200cm', 'RootMoist', 'SMLiq_0_10cm', 'SMLiq_10_40cm', 'SMLiq_40_100cm', 'SMLiq_100_200cm',
           'SMAvail_0_100cm', 'SMAvail_0_200cm', 'PotEvap', 'ECanop', 'TVeg', 'ESoil', 'SubSnow', 'CanopInt', 'ACond', 'CCond', 'RCS', 'RCT', 'RCQ', 'RCSOL', 'RSmin','RSMacr', 'LAI', 'GVEG', 'Streamflow']


schemat = StructType()
for i in columns:
  if i == "Date":
    schemat = schemat.add(i, StringType(), True)
  else:
    schemat = schemat.add(i, FloatType(), True)

nasa = spark.read.format('csv').option("header", True).schema(schemat).load('/content/drive/MyDrive/BigMess/NASA/NASA.csv')
nasa.show(5)

# usuwamy z danych kolumny odpowiadające za długość i szerokość geograficzną oraz datę
nasa = nasa.drop(*['lon', 'lat', 'Date'])
nasa.show(5)

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=nasa.columns, outputCol=vector_col, handleInvalid = 'skip')
nasa_vector = assembler.transform(nasa).select(vector_col)

# get correlation matrix
corr_matrix = Correlation.corr(nasa_vector, vector_col)

corr_matrix_pd = pd.DataFrame(np.array(corr_matrix.collect()[0][0].toArray()), columns=nasa.columns, index=nasa.columns)

# Tworzymy nasza heatmapę. Niestety funkcje pokroju sns.heatmap nie nadają się same w sobie do prezentacji tak dużych macierzy korelacji zatem potrzebne sa im pewne modyfikacje. Poniżej stowrzymy nasz wykres w taki sposób
# aby po kliknięciu na grafikę ulegała ona przybliżeniu i wyostrzeniu

def heatmap_plot(corr_matrix_pd, savefile_path: str, savefile_dpi: int = 300):

  corr_matrix_np = corr_matrix_pd.to_numpy()
  print(plt.get_backend())

  corr_matrix_np[np.triu_indices_from(corr_matrix_np)] = np.nan

  fig, ax = plt.subplots(figsize=(24, 18))

  hm = sns.heatmap(corr_matrix_np, cbar=True, vmin=-0.5, vmax=0.5,
                  fmt='.2f', annot_kws={'size': 8}, annot=True,
                  square=True, cmap=plt.cm.Blues)

  ticks = np.arange(corr_matrix_pd.shape[0]) + 0.5
  ax.set_xticks(ticks)
  ax.set_xticklabels(corr_matrix_pd.columns, rotation=90, fontsize=8)
  ax.set_yticks(ticks)
  ax.set_yticklabels(corr_matrix_pd.index, rotation=360, fontsize=8)

  ax.set_title('correlation matrix')
  plt.tight_layout()
  plt.savefig(savefile_path, dpi=savefile_dpi)


heatmap_plot(corr_matrix_pd, 'correlation_matrix.png')

#@title tabela najsilniej skorelowanych zmiennych

table = PrettyTable()
rows = [["LWdown", 'strumień promieniowania długofalowego (podczerwień) emitowanego przez atmosferę i kierowanego ku powierzchni ziemi', "0.66"], ["Qle", 'ilość ciepła przekazywanego w procesach, które nie pociągają za sobą zmiany temperatury, takich jak parowanie i kondensacja', "0.74"],
           ["Evap", 'Całkowita ewapotranspiracja: suma wody traconej z powierzchni ziemi do atmosfery przez parowanie i transpirację roślin', "0.74"],["Ecanop", 'Parowanie wody z baldachimu: proces, w którym woda paruje bezpośrednio z powierzchni liści i innych części roślin (tzw. "korony drzew")', "0.75"],
          ["TVeg", 'Transpiracja: Proces, w którym woda jest pobierana przez korzenie roślin, przemieszcza się przez roślinę i jest uwalniana do atmosfery przez aparaty szparkowe w liściach', "0.71"],
           ["CCond", 'Przewodność baldachimu: miara zdolności liści i innych części roślin do wymiany gazów i pary wodnej z atmosferą', "0.63"],["LAI", 'Wskaźnik pokrycia liściowego, indeks liściowy: stosunek całkowitej powierzchni liści roślin na jednostkę powierzchni gruntu', "0.61"]]
table.field_names = ["Nazwa zmiennej", "Objaśnienie zmiennej", "Współczynnik korelacji"]
for row in rows:
  table.add_row(row)
print(table)


