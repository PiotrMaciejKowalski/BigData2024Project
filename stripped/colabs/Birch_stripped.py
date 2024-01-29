#!/usr/bin/env python
# coding: utf-8

# # Grupowanie k-średnich z wykorzystaniem t-SNE (zamienione na algorymt Birch)

# ## Wczytywanie danych w sparku

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


get_ipython().system('git clone https://github.com/PiotrMaciejKowalski/BigData2024Project.git')
#%cd BigData2024Project
#!git checkout your-branch
#%cd ..


# In[3]:


get_ipython().system('chmod 755 /content/BigData2024Project/src/setup.sh')
get_ipython().system('/content/BigData2024Project/src/setup.sh')


# In[4]:


import sys
sys.path.append('/content/BigData2024Project/src')


# In[5]:


from start_spark import initialize_spark
initialize_spark()


# In[6]:


import pandas as pd
from pyspark.sql import SparkSession

from big_mess.loaders import default_loader, load_single_month, load_anotated, save_to_csv, preprocessed_loader


# In[7]:


spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()


# In[8]:


NASA_an = pd.read_csv('/content/drive/MyDrive/BigMess/NASA/NASA_an.csv',sep=';')
NASA_an["pustynia_i_step"] = NASA_an["pustynia"] + NASA_an["step"]


# ## Import bibliotek

# In[9]:


get_ipython().system('pip install datashader')
get_ipython().system('pip install holoviews hvplot colorcet')
get_ipython().system('pip install geoviews')


# In[10]:


from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from collections.abc import Iterable
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from typing import List, Union
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
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
from sklearn.cluster import Birch
import warnings
warnings.filterwarnings('ignore')


# ## Tworzenie funkcji

# In[11]:


def StandardScaling(df: pd.DataFrame, attributes: list) -> pd.DataFrame:
  """
  Funkcja służąca do przeskalowania wybranych atrybutów za pomocą funkcji StandardScaler, a następnie stworzenia nowej ramki danych z tylko przeskalowanymi atrybutami.
  Parametry:
  - df (DataFrame): Pandas DataFrame zawierająca co najmniej atrybuty,
  - attributes (str): atrybuty, które będziemy skalować.
  """
  scaled_data = StandardScaler().fit_transform(df[attributes])
  scaled_df = pd.DataFrame(scaled_data, columns=attributes)
  return scaled_df


# In[12]:


def MinMaxScaling(df: pd.DataFrame, attributes: list) -> pd.DataFrame:
  """
  Funkcja służąca do przeskalowania wybranych atrybutów za pomocą funkcji MinMaxScaler, a następnie stworzenia nowej ramki danych z tylko przeskalowanymi atrybutami.
  Parametry:
  - df (DataFrame): Pandas DataFrame zawierająca co najmniej atrybuty,
  - attributes (str): atrybuty, które będziemy skalować.
  """
  scaled_data = MinMaxScaler().fit_transform(df[attributes])
  scaled_df = pd.DataFrame(scaled_data, columns=attributes)
  return scaled_df


# In[13]:


'''
Funkcja jako argumenty bierze listę wartości określających granice przedziałów liczbowych, które
będą określać jak dla rozważanego parametru mają zmieniać się kolory punktów, których lista stanowi
drugi argument funkcji.
'''
def get_colormap(values: list, colors_palette: list, name = 'custom'):
    values = np.sort(np.array(values))
    values = np.interp(values, (values.min(), values.max()), (0, 1))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(name, list(zip(values, colors_palette)))
    return cmap


# In[14]:


def plot_map(df: pd.DataFrame, parameter_name: str, colormap: mpl.colors.LinearSegmentedColormap, title: str,
             point_size: int = 8, width: int = 800, height: int = 500, alpha: float = 1,
             bgcolor: str = 'white', colorbar_verbose: bool = False):

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
        colorbar=colorbar_verbose,
        toolbar='above',
        tools=['hover', 'wheel_zoom', 'reset'],
        alpha=alpha, # przezroczystość
        bgcolor=bgcolor
    )

    return hv.render(map_with_points)


# ## Przygotowanie danych

# Ponieważ poprzednie analizy pokazały, że grupowanie na podstawie miesięcy zimowych nie przynosi pozytywnych rezultatów, ograniczymy się do analizy miesięcy wiosennych, letnich i jesiennych.

# In[ ]:


nasa_202305 = load_single_month(spark, year=2023, month=5).toPandas()
nasa_202306 = load_single_month(spark, year=2023, month=6).toPandas()
nasa_202307 = load_single_month(spark, year=2023, month=7).toPandas()
nasa_202308 = load_single_month(spark, year=2023, month=8).toPandas()
nasa_202309 = load_single_month(spark, year=2023, month=9).toPandas()


# In[ ]:


list_of_df = [nasa_202305, nasa_202306, nasa_202307, nasa_202308, nasa_202309]


# In[ ]:


# Skalowanie danych
list_of_scaled_df = []
for dataframe in list_of_df:
  X = dataframe.loc[:,'Rainf':'SoilM_100_200cm']
  X_scaled = MinMaxScaling(StandardScaling(X, X.columns), X.columns)
  X_scaled = pd.concat([dataframe[["lon", "lat"]], X_scaled], axis=1)
  list_of_scaled_df.append(X_scaled)


# # Grupowanie algorytmem Birch

# **Birch** to oszczędzający pamięć algorytm do nauki online, stanowiący alternatywę dla MiniBatchKMeans. Konstruuje drzewiastą strukturę danych, w której centroidy klastrów są odczytywane z liścia.

# In[ ]:


# Grupowanie i dodanie kolumny "Cluster" do DataFramów
for scaled_df in list_of_scaled_df:
  birch_model = Birch(threshold=0.06, n_clusters=2)
  birch_model.fit(scaled_df.loc[:,'Rainf':'SoilM_100_200cm'])
  birch_results = birch_model.predict(scaled_df.loc[:,'Rainf':'SoilM_100_200cm'])
  scaled_df['Cluster'] = birch_results


# In[ ]:


for df, col_name in zip(list_of_scaled_df, list_of_df):
  colormap_cluster = get_colormap([0, max(df["Cluster"].values)], ['darkgreen', 'orange'])
  output_notebook()
  show(plot_map(df=df, parameter_name='Cluster', colormap=colormap_cluster, title=f"Dane z {col_name.loc[0,'Month']}.{col_name.loc[0,'Year']}", alpha=0.5))


# In[ ]:


listed_of_merged_df = []
for df in list_of_scaled_df:
  merged_df = df[['lon', 'lat', 'Cluster']].merge(NASA_an[['lon', 'lat', 'pustynia', 'pustynia_i_step']], left_on=['lon','lat'], right_on = ['lon','lat'], how='inner')
  listed_of_merged_df.append(merged_df)


# In[ ]:


predykcje_df = pd.DataFrame({"Predykcje 10.2022": listed_of_merged_df[0]['Cluster'],
                       "Predykcje 04.2023": listed_of_merged_df[1]['Cluster'],
                       "Predykcje 05.2023": listed_of_merged_df[2]['Cluster'],
                       "Predykcje 06.2023": listed_of_merged_df[3]['Cluster'],
                       "Predykcje 07.2023": listed_of_merged_df[4]['Cluster'],
                       "Predykcje 08.2023": listed_of_merged_df[5]['Cluster'],
                       "Predykcje 09.2023": listed_of_merged_df[6]['Cluster']})


# In[ ]:


wyniki = pd.DataFrame({"Name": ["Predykcje 10.2022","Predykcje 04.2023","Predykcje 05.2023","Predykcje 06.2023","Predykcje 07.2023","Predykcje 08.2023","Predykcje 09.2023"]})
wyniki["Accuracy [%]"] = wyniki["Name"].apply(lambda name: round(accuracy_score(listed_of_merged_df[0]["pustynia_i_step"], predykcje_df[name])*100, 2))
wyniki["Precision [%]"] = wyniki["Name"].apply(lambda name: round(precision_score(listed_of_merged_df[0]["pustynia_i_step"], predykcje_df[name])*100, 2))
wyniki["Recall [%]"] = wyniki["Name"].apply(lambda name: round(recall_score(listed_of_merged_df[0]["pustynia_i_step"], predykcje_df[name])*100, 2))
wyniki["F1-Score [%]"] = wyniki["Name"].apply(lambda name: round(f1_score(listed_of_merged_df[0]["pustynia_i_step"], predykcje_df[name])*100, 2))
wyniki.sort_values(by=['Accuracy [%]'], ascending=False)


# # Tunning hiperparametrów modelu trenowanego na danych z lipca 2023

# In[19]:


X = nasa_202307.loc[:,'Rainf':'SoilM_100_200cm']
X_scaled = MinMaxScaling(StandardScaling(X, X.columns), X.columns)
X_scaled = pd.concat([nasa_202307[["lon", "lat"]], X_scaled], axis=1)
merged_df = X_scaled.merge(NASA_an[['lon', 'lat', 'pustynia', 'pustynia_i_step']], left_on=['lon','lat'], right_on = ['lon','lat'], how='inner')


# In[20]:


for threshold in (0.03, 0.04):
  for branching_factor in (50,55,60):
    birch_model = Birch(n_clusters=2, threshold=threshold, branching_factor=branching_factor)
    birch_model.fit(X_scaled.loc[:,'Rainf':'SoilM_100_200cm'])
    birch_results = birch_model.predict(merged_df.loc[:,'Rainf':'SoilM_100_200cm'])
    print("Threshold: ", threshold, "Branching factor: ", branching_factor)
    print(np.unique(birch_results, return_counts=True))
    print("Accuracy: ", round(accuracy_score(merged_df["pustynia_i_step"], birch_results)*100, 2))
    print("Precision: ", round(precision_score(merged_df["pustynia_i_step"], birch_results)*100, 2))
    print("Recall: ", round(recall_score(merged_df["pustynia_i_step"], birch_results)*100, 2))
    print("F1-score: ", round(f1_score(merged_df["pustynia_i_step"], birch_results)*100, 2))
    print("- - - - - - - - - - - - - -")


# In[21]:


birch_model = Birch(n_clusters=2, threshold=0.04, branching_factor=55)
birch_model.fit(X_scaled.loc[:,'Rainf':'SoilM_100_200cm'])
birch_results = birch_model.predict(merged_df.loc[:,'Rainf':'SoilM_100_200cm'])


# In[22]:


merged_df["Cluster"] = birch_results


# In[24]:


colormap_cluster = get_colormap([0, max(birch_results)], ['darkgreen', "orange"])
output_notebook()
show(plot_map(df=merged_df, parameter_name='Cluster', colormap=colormap_cluster, title="Wyniki na zbiorze anotowanym", alpha=0.5))


# # Wnioski

# * Grupowanie k-średnich z wykorzystaniem t-SNE nie powiodło się sukcesem ze
# względu na duży zbiór danych. Po zastosowaniu algorytmu dla różnych wartości preplexity oraz learning_rate algorytm nie rozdzielił danych, co uniemożliwiło grupowanie.
# * Grupowanie hierarchiczne dla parametrów linkage: "ward", "average", "complete" nie powiodło się ponieważ na Colabie jest zbyt mała pamięć RAM i za każdym razem usuwało sesję. Grupowanie to zadziałało tylko dla linkage = "single" jednak wyniki na zbiorze anotowanym były gorsze od tych powyżej.
# * Grupowanie algorytmem Birch nie zrobiło szału. Wyniki podobne lub trochę gorsze od zwykłego algorytmu k-średnich.

# # Pomysły, które nie zakończyły się sukcesem

# In[ ]:


def do_kmeans_and_return_df_with_cluster_column(df: pd.DataFrame, scaled_df: pd.DataFrame, n_clusters: int, init: str, random_state: int) -> pd.DataFrame:
  """
  Funkcja wykonuje grupowanie k-średnich dla n_clusters klastrów oraz tworzy nową kolumnę z predykcjami algorytmu k-średnich w ramce danych df.
  Parametry:
  - df (DataFrame): Pandas DataFrame zawierająca co najmniej te same kolumny co scaled_df,
  - scaled_df (DataFrame): Pandas DataFrame zawierająca przeskalowane kolumny, na podstawie których dokonywane jest grupowanie,
  - n_clusters (int): maksymalna liczba klastrów,
  - random_state (int): ziarno losowości.
  """
  kmeans = KMeans(n_clusters = n_clusters, init = init, random_state = random_state)
  kmeans.fit(scaled_df)
  pred = kmeans.predict(scaled_df)
  df['cluster'] = pred
  return df


# In[ ]:


def plot_2d_tsne_for_three_different_perplexities(list_of_perplexities: Iterable[int], scaled_df: pd.DataFrame, czy_step_to_pustynia: bool=False) -> None:
  """
  Funkcja tworzy trzy wykresy dla różnych wartości perplexity.
  Parametry:
  - list_of_perplexities - lista wartości perplexity dla których chcemy stworzyć wykresy,
  - scaled_df - ramka danych z już przeskalowanymi wartościami na podstawie której wykonujemy algorytm t-SNE,
  - df_with_target - ramka danych w której znajdują się kolumny "pustynia" i "step",
  - czy_step_to_pustynia - wartość logiczna, gdy TRUE to traktujemy step jako pustynię.
  """
  fig, axes = plt.subplots(ncols=3, figsize=(18, 6))

  scaled_df_train = scaled_df.loc[:,'Rainf':'SoilM_100_200cm']

  for perplexity, ax in zip(list_of_perplexities, axes.flat):
    tsne = TSNE(n_components = 2, perplexity = perplexity, n_jobs=-1, random_state=2024)
    tsne_results = tsne.fit_transform(scaled_df_train)

    df = pd.DataFrame()
    df['dim 1'] = tsne_results[:,0]
    df['dim 2'] = tsne_results[:,1]
    if czy_step_to_pustynia:
      df['Cluster'] = scaled_df["pustynia_i_step"]
    else:
      df['Cluster'] = scaled_df["pustynia"]
    sns.scatterplot(x="dim 1", y="dim 2", hue="Cluster", palette = ["darkgreen", "orange", "blue"], data=df, legend="full", ax=ax).set(title=f'Perplexity: {perplexity}, Divergence: {round(tsne.kl_divergence_,3)}')


# In[ ]:


def plot_3d_tsne(perplexity: float, scaled_df: pd.DataFrame, original_df: pd.DataFrame, anotated_df: pd.DataFrame, czy_step_to_pustynia: bool=False) -> None:
  """
  Funkcja tworzy wykres 3d dla wybranej wartości perplexity.
  Parametry:
  - list_of_perplexities - lista wartości perplexity dla których chcemy stworzyć wykresy,
  - scaled_df - ramka danych z już przeskalowanymi wartościami na podstawie której wykonujemy algorytm t-SNE,
  - df_with_target - ramka danych w której znajdują się kolumny "pustynia", "step", "lon" i "lat",
  - czy_step_to_pustynia - wartość logiczna, gdy TRUE to traktujemy step jako pustynię.
  """
  tsne = TSNE(n_components = 3, perplexity = perplexity)
  tsne_results = tsne.fit_transform(scaled_df)

  df_merged_with_anotated = original_df.merge(anotated_df[['lon', 'lat', 'pustynia', 'step', 'pustynia_i_step']], left_on=['lon','lat'], right_on = ['lon','lat'], how='left')
  df_merged_with_anotated.fillna(2, inplace=True)

  df = pd.DataFrame()
  df['dim 1'] = tsne_results[:,0]
  df['dim 2'] = tsne_results[:,1]
  df['dim 3'] = tsne_results[:,2]
  if czy_step_to_pustynia:
    df['Cluster'] = df_merged_with_anotated["pustynia_i_step"]
  else:
    df['Cluster'] = df_merged_with_anotated["pustynia"]

  fig = go.Figure(data=[go.Scatter3d(
    x=df['dim 1'],
    y=df['dim 2'],
    z=df['dim 3'],
    mode="markers",
    marker=dict(color=df['Cluster'], colorscale=["darkgreen", "orange"]),
    text = [f"Współrzędne: ({lat}, {lon})" for lat, lon in df_merged_with_anotated[["lat", "lon"]].values],
    hoverinfo='text'
  )])
  fig.update_layout(scene = dict(
                    xaxis_title='dim 1',
                    yaxis_title='dim 2',
                    zaxis_title='dim 3'))
  fig.update_layout(scene=dict(xaxis_showspikes=False,
                             yaxis_showspikes=False))
  fig.show()


# In[ ]:


def do_kmeans_with_tsne_return_preds(list_of_scaled_df: List, n_components: int=2, perplexity:float=30, learning_rate: Union[float, str]="auto", n_clusters: int=2, random_state: int=123) -> List:
  warnings.filterwarnings('ignore')
  list_of_preds = []
  for scaled_df in list_of_scaled_df:
    tsne = TSNE(n_components = n_components, perplexity = perplexity, learning_rate=learning_rate)
    tsne_results = tsne.fit_transform(scaled_df)

    kmeans = KMeans(n_clusters = n_clusters, init = "k-means++", random_state = random_state)
    kmeans.fit(tsne_results)
    pred = kmeans.predict(tsne_results)
    list_of_preds.append(pred)
  return list_of_preds


# In[ ]:


hierarchical_cluster_ward = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
labels_ward = hierarchical_cluster_ward.fit_predict(X_scaled.loc[:,'Rainf':'SoilM_100_200cm'])


# In[ ]:


nasa_202307['Cluster'] = birch_result


# In[ ]:


np.unique(labels_ward, return_counts=True)


# In[ ]:


colormap_cluster = get_colormap([0, max(nasa_202307['Cluster'].values)], ['darkgreen', "orange"])
output_notebook()
show(plot_map(df=nasa_202307, parameter_name='Cluster', colormap=colormap_cluster, title="Predykce", alpha=0.5))


# In[ ]:


nasa_anotated_500_202307 = nasa_202307.merge(NASA_sample_an, left_on=['lon','lat'], right_on = ['lon','lat'], how='inner')
nasa_anotated_500_202307.head()


# In[ ]:


list_of_df = [nasa_202210, nasa_202304, nasa_202305, nasa_202306, nasa_202307, nasa_202308, nasa_202309]


# In[ ]:


X = nasa_202307.loc[:,'Rainf':'SoilM_100_200cm']
X_scaled = MinMaxScaling(StandardScaling(X, X.columns), X.columns)
X_scaled = pd.concat([nasa_202307[["lon", "lat"]], X_scaled], axis=1)
X_scaled.head()


# In[ ]:


X = nasa_anotated_500_202307.loc[:,'Rainf':'SoilM_100_200cm']
X_scaled = MinMaxScaling(StandardScaling(X, X.columns), X.columns)
X_scaled = pd.concat([nasa_anotated_500_202307[["lon", "lat", "pustynia", "step", "pustynia_i_step"]], X_scaled], axis=1)
X_scaled.head()


# In[ ]:


plot_2d_tsne_for_three_different_perplexities([10,30,50], X_scaled, czy_step_to_pustynia=True)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


tsne = TSNE(n_components = 2, perplexity = 50, learning_rate=10000, random_state=2024, n_jobs=-1)
tsne_results = tsne.fit_transform(X_scaled.loc[:,'Rainf':'SoilM_100_200cm'])


# In[ ]:


df = pd.DataFrame()
df['dim 1'] = tsne_results[:,0]
df['dim 2'] = tsne_results[:,1]
sns.scatterplot(x="dim 1", y="dim 2", palette = ["darkgreen", "orange", "blue"], data=df, legend="full")


# In[ ]:


df = pd.DataFrame()
df['dim 1'] = tsne_results[:,0]
df['dim 2'] = tsne_results[:,1]
df['Cluster'] = X_scaled["pustynia_i_step"]


# In[ ]:


clf = LogisticRegression(random_state=0).fit(df[["dim 1", "dim 2"]], nasa_anotated_500_202307['pustynia_i_step'])


# In[ ]:


df['Cluster pred'] = clf.predict(df[["dim 1", "dim 2"]])


# In[ ]:


sns.scatterplot(x="dim 1", y="dim 2", hue="Cluster", palette = ["darkgreen", "orange", "blue"], data=df, legend="full").set(ylim=(min(df['dim 2']-1), max(df['dim 2']+1)), xlim=(min(df['dim 1']-1), max(df['dim 1']+1)), title="Regresja logistyczna")
b = clf.intercept_[0]
w1, w2 = clf.coef_.T

c = -b/w2
m = -w1/w2

xmin, xmax = min(df['dim 1']-1), max(df['dim 1']+1)
ymin, ymax = min(df['dim 2']-1), max(df['dim 2']+1)
xd = np.array([xmin, xmax])
yd = m*xd + c
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, ymin, color='tab:green', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(clf.predict(df), nasa_anotated_500_202307['pustynia_i_step']))


# Do grupowania będziemy używać następujących wskaźników:
# 
# *   **Rainf** - wskaźnik opadów deszczu,
# *   **Evap** - wskaźnik całkowitej ewapotranspiracji,
# *   **AvgSurfT** - wskaźnik średniej temperatury powierzchni ziemi,
# *   **Albedo** - wskaźnik albedo,
# *   **SoilT_40_100cm** - wskaźnik temperatury gleby w warstwie o głębokości od 40 do 100 cm,
# *   **GVEG** - wskaźnik roślinności,
# *   **PotEvap** - wskaźnik potencjalnej ewapotranspiracji,
# *   **RootMoist** - wilgotność gleby w strefie korzeniowej (parowanie, które miałoby miejsce, gdyby dostępne było wystarczające źródło wody),
# *   **SoilM_100_200cm** - wilgotność gleby w warstwie o głębokości od 100 do 200 cm.

# In[ ]:


# Elementy list_of_scaled_df są w kolejności chronologicznej tzn. list_of_scaled_df[0] to dane z 10.2022, a list_of_scaled_df[11] to dane z 09.2023.
list_of_scaled_df = []
for dataframe in list_of_df:
  X = dataframe.loc[:,'Rainf':'SoilM_100_200cm']
  X_scaled = MinMaxScaling(StandardScaling(X, X.columns), X.columns)
  X_scaled = pd.concat([dataframe[["lon", "lat"]], X_scaled], axis=1)
  list_of_scaled_df.append(X_scaled)


# ## Implementacja algorytmu k-średnich z wykorzystaniem t-SNE dla danych zanotowanych

# Na początku spróbujmy dobrać optymalny hiperparametr *perplexity*. W tym celu przedstawmy na kilku wykresach wyniki algorytmu t-SNE dla różnych wartości perplexity.

# **Przypadek 1.** step nie jest pustynią.

# In[ ]:


plot_2d_tsne_for_three_different_perplexities([200,100,300], list_of_scaled_df[6], nasa_anotated)


# In[ ]:


plot_3d_tsne(30, list_of_scaled_df[11], df_202309, False)


# **Przypadek 2.** step jest pustynią.

# In[ ]:


plot_2d_tsne_for_nine_different_perplexities([10,30,40,50,60,70,80,100,130], list_of_scaled_df[11], df_202309, True)


# In[ ]:


plot_3d_tsne(30, list_of_scaled_df[11], df_202309, True)


# Spróbujmy użyć wartość perplexity = 50.

# In[ ]:


predykcje = do_kmeans_with_tsne_return_preds(list_of_scaled_df, perplexity=50, n_components=3)


# In[ ]:


predykcje_df = pd.DataFrame({"Predykcje 10.2022": predykcje[0],
                       "Predykcje 04.2023": predykcje[6],
                       "Predykcje 05.2023": predykcje[7],
                       "Predykcje 06.2023": predykcje[8],
                       "Predykcje 07.2023": predykcje[9],
                       "Predykcje 08.2023": predykcje[10],
                       "Predykcje 09.2023": predykcje[11]})


# In[ ]:


predykcje_df.head(7)


# In[ ]:


#predykcje_df["Predykcje 08.2023"] = predykcje_df["Predykcje 08.2023"].apply(lambda x: 1 if x==0 else 1)
#predykcje_df["Predykcje 09.2023"] = predykcje_df["Predykcje 09.2023"].apply(lambda x: 1 if x==0 else 1)

#predykcje_df["Predykcje 10.2022"] = predykcje_df["Predykcje 10.2022"].apply(lambda x: 1 if x==0 else 1)
#predykcje_df["Predykcje 05.2023"] = predykcje_df["Predykcje 05.2023"].apply(lambda x: 1 if x==0 else 1)
#predykcje_df["Predykcje 02.2023"] = predykcje_df["Predykcje 02.2023"].apply(lambda x: 1 if x==0 else 1)


# ## Wyniki

# Tak wyglądają Stany Zjednoczone.

# In[ ]:


display(IFrame("https://www.google.com/maps/embed?pb=!1m14!1m12!1m3!1d13982681.959428234!2d-98.66341902257437!3d38.39997874427714!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!5e1!3m2!1spl!2spl!4v1703000232420!5m2!1spl!2spl", '800px', '500px'))


# **Przypadek 1.** step nie jest pustynią

# In[ ]:


colormap_cluster = get_colormap([0, max(df_202309['pustynia'].values)], ['darkgreen', "orange"])
output_notebook()
show(plot_map(df=df_202309, parameter_name='pustynia', colormap=colormap_cluster, title="Dane zanotowane, step nie jest pustynią", alpha=0.5))


# In[ ]:


wyniki = pd.DataFrame({"Name": ["Predykcje 10.2022","Predykcje 04.2023","Predykcje 05.2023","Predykcje 06.2023","Predykcje 07.2023","Predykcje 08.2023","Predykcje 09.2023"]})
wyniki["Accuracy [%]"] = wyniki["Name"].apply(lambda name: round(accuracy_score(df_202309["pustynia"], predykcje_df[name])*100, 2))
wyniki["Precision [%]"] = wyniki["Name"].apply(lambda name: round(precision_score(df_202309["pustynia"], predykcje_df[name])*100, 2))
wyniki["Recall [%]"] = wyniki["Name"].apply(lambda name: round(recall_score(df_202309["pustynia"], predykcje_df[name])*100, 2))
wyniki["F1-Score [%]"] = wyniki["Name"].apply(lambda name: round(f1_score(df_202309["pustynia"], predykcje_df[name])*100, 2))
wyniki.sort_values(by=['Accuracy [%]'], ascending=False)


# **Przypadek 2.** step jest pustynią

# In[ ]:


colormap_cluster = get_colormap([0, max(df_202309['pustynia_i_step'].values)], ['darkgreen', "orange"])
output_notebook()
show(plot_map(df=df_202309, parameter_name='pustynia_i_step', colormap=colormap_cluster, title="Dane zanotowane, step jest pustynią", alpha=0.5))


# In[ ]:


wyniki = pd.DataFrame({"Name": ["Predykcje 10.2022","Predykcje 04.2023","Predykcje 05.2023","Predykcje 06.2023","Predykcje 07.2023","Predykcje 08.2023","Predykcje 09.2023"]})
wyniki["Accuracy [%]"] = wyniki["Name"].apply(lambda name: round(accuracy_score(df_202309["pustynia_i_step"], predykcje_df[name])*100, 2))
wyniki["Precision [%]"] = wyniki["Name"].apply(lambda name: round(precision_score(df_202309["pustynia_i_step"], predykcje_df[name])*100, 2))
wyniki["Recall [%]"] = wyniki["Name"].apply(lambda name: round(recall_score(df_202309["pustynia_i_step"], predykcje_df[name])*100, 2))
wyniki["F1-Score [%]"] = wyniki["Name"].apply(lambda name: round(f1_score(df_202309["pustynia_i_step"], predykcje_df[name])*100, 2))
wyniki.sort_values(by=['Accuracy [%]'], ascending=False)


# In[ ]:


for dataframe, col_name in zip(list_of_df, predykcje_df.columns):
  dataframe["Cluster"] = predykcje_df[col_name]
  colormap_cluster = get_colormap([0, max(dataframe["Cluster"].values)], ['darkgreen', 'orange'])
  output_notebook()
  show(plot_map(df=dataframe, parameter_name='Cluster', colormap=colormap_cluster, title=f"{col_name}, k-średnich z t-SNE", alpha=0.5))

