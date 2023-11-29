"""
<a href="https://colab.research.google.com/github/PiotrMaciejKowalski/BigData2024Project/blob/Narzdzie-do-wizualizacji-na-mapach/mapy_datashader.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

!pip install datashader


!pip install holoviews hvplot colorcet


!pip install geoviews


import datashader as ds
import datashader.transfer_functions as tf
import numpy as np
import colorcet as cc
import holoviews as hv
# from holoviews.element.tiles import EsriImagery
from holoviews.operation.datashader import datashade
import pandas as pd
import geoviews as gv
# import hvplot.pandas
import geoviews.tile_sources as gts
from holoviews import opts
import matplotlib as mpl
import matplotlib.pyplot as plt


"""
Wczytujemy stworzony wcześniej plik z danymi na jeden miesiąc:
"""

df = pd.read_csv('/content/nasa_dec22_loc.csv')
df


"""
Definiujemy funkcję, która umożliwi tworzenie własnych zakresów kolorów użytych do map:
"""

'''
Funkcja jako argumenty bierze listę wartości określających granice przedziałów liczbowych, które
będą określać jak dla rozważanego parametru mają zmieniać się kolory punktów, których lista stanowi
drugi argument funkcji.
'''
def get_colormap(values, colors_palette, name = 'custom'):
    values = np.sort(np.array(values))
    values = np.interp(values, (values.min(), values.max()), (0, 1))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(name, list(zip(values, colors_palette)))
    return cmap


"""
Tworzymy pierwszą mapę, w tym wypadku dla parametru GVEG związanego z poziomem roślinności:
"""

colormap_GVEG = get_colormap([0, max(df.GVEG.values)], ['yellow', 'darkgreen'])

gdf_GVEG = gv.Points(df, ['lon', 'lat'], ['GVEG']) # obiekt zawierający punkty

tiles_GVEG = gts.OSM # wybieramy mapę tła, w tym wypadku OpenStreetMap

# łączymy mapę tła z punktami i ustawiamy wybrane parametry wizualizacji
map_with_points_gts_GVEG = tiles_GVEG * gdf_GVEG.opts(
    color = 'GVEG',
    cmap = colormap_GVEG,
    size = 8,
    width = 900,
    height = 600,
    colorbar = True,
    toolbar = 'above',
    tools = ['hover', 'wheel_zoom', 'reset'],
    alpha = 0.5 # przezroczystość
    )

map_with_points_gts_GVEG.opts(bgcolor='white')

# zapisujemy mapę do pliku .html
hv.save(map_with_points_gts_GVEG.opts(width=900, height=600, bgcolor='white', xaxis=None, yaxis=None), 'output_map_osm_GVEG.html')
#hv.save(map_with_points_gts_GVEG.opts(width=900, height=600, xaxis=None, yaxis=None), 'output_map_osm_GVEG_3.html')


"""
Druga mapa, tym razem dla parametru Rainf określającego poziom opadów deszczu:
"""

colormap_Rainf = get_colormap([0, max(df.Rainf.values)/2, max(df.Rainf.values)], ['white', 'blue', 'black'])

gdf_Rainf = gv.Points(df, ['lon', 'lat'], ['Rainf'])

tiles_Rainf = gts.OSM

map_with_points_gts_Rainf = tiles_Rainf * gdf_Rainf.opts(
    color = 'Rainf',
    cmap = colormap_Rainf,
    size = 5,
    width = 900,
    height = 600,
    colorbar = True,
    toolbar = 'above',
    tools = ['hover', 'wheel_zoom', 'reset'],
#    alpha = 0.75
    )

map_with_points_gts_Rainf.opts() #bgcolor='black')

# hv.save(map_with_points_gts_Rainf.opts(width=900, height=600, bgcolor='black', xaxis=None, yaxis=None), 'output_map_osm_Rainf.html')
hv.save(map_with_points_gts_Rainf.opts(width=900, height=600, xaxis=None, yaxis=None), 'output_map_osm_Rainf.html')


"""
Trzecia mapa dla AvgSurfT, tj. średniej temperatury powierzchni:
"""

colormap_AvgSurfT = 'cet_rainbow_bgyr_10_90_c83'

gdf_AvgSurfT = gv.Points(df, ['lon', 'lat'], ['AvgSurfT'])

tiles_AvgSurfT = gts.OSM

map_with_points_gts_AvgSurfT = tiles_AvgSurfT * gdf_AvgSurfT.opts(
    color = 'AvgSurfT',
    cmap = colormap_AvgSurfT
    size = 5,
    width = 900,
    height = 600,
    colorbar = True,
    toolbar = 'above',
    tools = ['hover', 'wheel_zoom', 'reset'],
    alpha = 0.7 # przezroczystość
    )

map_with_points_gts_AvgSurfT.opts(bgcolor='white')

hv.save(map_with_points_gts_AvgSurfT.opts(width=900, height=600, bgcolor='white', xaxis=None, yaxis=None), 'output_map_osm_AvgSurfT.html')


