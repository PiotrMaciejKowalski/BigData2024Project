pip install folium


pip install plotly-express


import pandas as pd
import requests
import folium
from folium.plugins import HeatMap, MiniMap
import numpy as np
import branca.colormap as cm
import plotly.express as px


"""
Wczytajmy utworzony wcześniej podzbiór danych:
"""

df = pd.read_csv('nasa_dec22_loc.csv')


"""
# Folium
"""

"""
## Interaktywne mapy z wykorzystaniem OpenStreetMap
"""

"""
User guide odnośnie korzystania z biblioteki Folium i poszczególnych parametrów używanych w wizualizacjach: https://python-visualization.github.io/folium/latest/user_guide.html
"""

"""
### Heatmap
"""

"""
Zastosowanie gotowej funkcji HeatMap.
W tym oraz kolejnych przykładach pobierana jest lista zawierająca zmienne dotyczące każdego z punktów, gdzie pierwsze dwie określają lokalizację, a trzecia wybrany parametr, w tym wypadku Rainf.
"""

# lokalizacja, która wyświetlana jest domyślnie po utworzeniu mapy
heatmap_Rainf = folium.Map(location = [38.27312, -98.5821872], zoom_start = 4)

MiniMap(toggle_display=True).add_to(heatmap_Rainf) # dodanie minimapy

heatmap_Rainf_param = HeatMap(list(zip(df.lat.values, df.lon.values, df.Rainf.values)),
                              min_opacity=0.3,
                              radius=20,
                              blur=15
                             )

heatmap_Rainf.add_child(heatmap_Rainf_param)


"""
W tym wypadku, prawdopodobnie ze względu na dużą liczbę punktów naniesionych na mapę, finalny efekt nie jest do końca satysfakcjonujący, zwłaszcza przy przybliżaniu.
"""

"""
### Circle
"""

"""
Zastosowanie funkcji Circle.
W pierwszym przypadku punkty będą kolorowane określonymi barwami, w zależności od tego czy ich wartości wpadają w dany przedział. W przykładzie przedziały zdefiniujemy przez podzielenie całego zakresu wartości kwartylami. Do określenia gamy kolorystycznej stosujemy funkcję StepColormap. W przypadku pominięcia parametru index zakres wartości dla każdego koloru zostanie utworzony automatycznie przez podzielenie całego przedziału na równe części.
"""

min_val = min(df.Rainf.values)
q1 = np.quantile(df.Rainf.values, 0.25)
q2 = np.quantile(df.Rainf.values, 0.5)
q3 = np.quantile(df.Rainf.values, 0.75)
max_val = max(df.Rainf.values)

step_colormap = cm.StepColormap(["green", "yellow", "orange", "red"],
                                vmin=min_val, vmax=max_val,
                                index=[min_val, q1, q2, q3, max_val])

# lokalizacja, która wyświetlana jest domyślnie po utworzeniu mapy
dotted_map_Rainf_step = folium.Map(location = [38.27312, -98.5821872], zoom_start = 4)

# umieszczenie odpowiednio pokolorowanych punktów na mapie
for loc, val in list(zip(zip(df.lat.values, df.lon.values), df.Rainf.values)):
    folium.Circle(
        location=loc,
        radius=8,
        fill=True,
        color=step_colormap(val),
#         popup=f'Location: {loc}, Rainfall: {val:.2f}' # określenie informacji wyświetlanej po kliknięciu na punkt
    ).add_to(dotted_map_Rainf_step)

MiniMap(toggle_display=True).add_to(dotted_map_Rainf_step) # dodanie minimapy

dotted_map_Rainf_step


"""
Teraz do określenia gamy kolorystycznej użyjemy funkcji LinearColormap, żeby zastosować płynniejsze przejścia między kolorami.
"""

colormap = cm.LinearColormap(colors=['green', 'yellow', 'orange', 'red'], vmin=min(df.Rainf.values), vmax=max(df.Rainf.values))

# lokalizacja, która wyświetlana jest domyślnie po utworzeniu mapy
dotted_map_Rainf_smooth = folium.Map(location=[38.27312, -98.5821872], zoom_start=4)

for loc, temp in list(zip(zip(df.lat.values, df.lon.values), df.Rainf.values)):
    folium.Circle(
        location=loc,
        radius=8,
        fill=True,
        color=colormap(temp),
        fill_opacity=0.2,
#         popup=f'Location: {loc}, Rainfall: {val:.2f}' # określenie informacji wyświetlanej przy najechaniu na punkt
    ).add_to(dotted_map_Rainf_smooth)

MiniMap(toggle_display=True).add_to(dotted_map_Rainf_smooth) # dodanie minimapy

dotted_map_Rainf_smooth.add_child(colormap) # dodanie legendy


"""
Każdą z utworzonych map możemy zapisać do pliku html:
"""

heatmap_Rainf.save('heatmap_Rainf.html')
dotted_map_Rainf_step.save('dotted_map_Rainf_step.html')
dotted_map_Rainf_smooth.save('dotted_map_Rainf_smooth.html')




