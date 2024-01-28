!pip install datashader
!pip install holoviews hvplot colorcet
!pip install geoviews

import pandas as pd
import matplotlib as mpl
import holoviews as hv
import geoviews as gv
import geoviews.tile_sources as gts
from bokeh.plotting import show, output_notebook
from google.colab import drive
import os
drive.mount("/content/drive")

def plot_map(df: pd.DataFrame, parameter_name: str, colormap: mpl.colors.LinearSegmentedColormap,
             title: str, point_size: int = 8, width: int = 900, height: int = 600, alpha: float = 1,
             bgcolor: str = 'white'):

    gdf = gv.Points(df, ['lon', 'lat'], [parameter_name]) # obiekt zawierający punkty
    tiles = gts.OSM # wybór mapy tła, w tym wypadku OpenStreetMap

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
    return hv.render(map_with_points)

NASA_an = pd.read_csv('/content/drive/MyDrive/BigMess/NASA/NASA_an.csv', sep=';')

NASA_an

pustynia = NASA_an['pustynia'].sum()
pustynia_procent = pustynia / len(NASA_an)

print(f"Etykietę 'pustynia' otrzymało {pustynia} lokalizacji, co stanowi {pustynia_procent*100:.2f}% wszystkich zaanotowanych obszarów.")

step = NASA_an['step'].sum()
step_procent = step / len(NASA_an)

print(f"Etykietę 'step' otrzymało {step} lokalizacji, co stanowi {step_procent*100:.2f}% wszystkich zaanotowanych obszarów.")

inne = len(NASA_an) - pustynia - step
inne_procent = inne / len(NASA_an)

print(f"Za żaden z rozważanych terenów zostało uznane {inne} lokalizacji, co stanowi {inne_procent*100:.2f}% wszystkich zaanotowanych obszarów.")

NASA_an_points = NASA_an.copy()
NASA_an_points['klasyfikacja'] = 'niepustynia'
NASA_an_points.loc[NASA_an_points['pustynia'] == 1, 'klasyfikacja'] = 'pustynia'
NASA_an_points.loc[NASA_an_points['step'] == 1, 'klasyfikacja'] = 'step'
output_notebook()
show(plot_map(df = NASA_an_points, parameter_name = 'klasyfikacja',
              colormap = dict(zip(['pustynia', 'niepustynia', 'step'], ['yellow', 'green', 'orange'])),
              title = "Zaanotowane punkty", point_size = 6, alpha = 0.7))
