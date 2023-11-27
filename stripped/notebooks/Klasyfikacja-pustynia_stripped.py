import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


NASA = pd.read_csv('sampled_NASA_200k.csv')


NASA.head(5)


print("Lon Range:", NASA['lon'].min(), NASA['lon'].max())
print("Lat Range:", NASA['lat'].min(), NASA['lat'].max())


"""
##### Wybranie ciepłych miesięcy, aby otrzymać realny obraz występującej roślinności
"""

NASA['Date'] = NASA['Date'].astype(str)
NASA_lato = NASA[NASA['Date'].str[-2:].isin(["05", "06", "07", "08", "09", "10"])]


"""
##### Kolumny, które mogą mieć największy wpływ na pustynnienie (wg ustaleń Mariusza po przeprowadzonym researchu)
"""

selected_columns = ["lon", "lat", "Rainf", "Evap", "AvgSurfT", "Albedo", "SoilT_40_100cm", "GVEG"]
NASA_lato = NASA_lato[selected_columns].dropna()


NASA_lato.head(5)


"""
# CD - Chihuahuan Desert
# CP - Colorado Plateau
# GBD - Great Basin Desert
"""

"""
##### Obszary pustynne
"""

CD = NASA_lato[(NASA_lato['lon'] >= -104) & (NASA_lato['lon'] <= -102) & (NASA_lato['lat'] >= 30) & (NASA_lato['lat'] <= 31)]
CP = NASA_lato[(NASA_lato['lon'] >= -110.5) & (NASA_lato['lon'] <= -108.5) & (NASA_lato['lat'] >= 39) & (NASA_lato['lat'] <= 40.5)]
GBD = NASA_lato[(NASA_lato['lon'] >= -116) & (NASA_lato['lon'] <= -114) & (NASA_lato['lat'] >= 40) & (NASA_lato['lat'] <= 41.5)]


"""
##### Obszary pustynno - niepustynne
"""

CD_i_niepustynia = NASA_lato[(NASA_lato['lon'] >= -106.5) & (NASA_lato['lon'] <= -104.5) & (NASA_lato['lat'] >= 32.5) & (NASA_lato['lat'] <= 33.5)]
CP_i_niepustynia = NASA_lato[(NASA_lato['lon'] >= -109) & (NASA_lato['lon'] <= -107) & (NASA_lato['lat'] >= 37.5) & (NASA_lato['lat'] <= 39)]
GBD_i_niepustynia = NASA_lato[(NASA_lato['lon'] >= -115) & (NASA_lato['lon'] <= -113) & (NASA_lato['lat'] >= 42.5) & (NASA_lato['lat'] <= 44)]


"""
##### Obszary niepustynne
"""

niepustynia_przy_CD = NASA_lato[(NASA_lato['lon'] >= -109.5) & (NASA_lato['lon'] <= -107.5) & (NASA_lato['lat'] >= 33) & (NASA_lato['lat'] <= 34)]
niepustynia_przy_CP = NASA_lato[(NASA_lato['lon'] >= -107) & (NASA_lato['lon'] <= -105) & (NASA_lato['lat'] >= 39) & (NASA_lato['lat'] <= 40.5)]
niepustynia_przy_GBD = NASA_lato[(NASA_lato['lon'] >= -124) & (NASA_lato['lon'] <= -122) & (NASA_lato['lat'] >= 39.5) & (NASA_lato['lat'] <= 41)]


"""
# Wizualizacje + wstępne ustalenia parametrów
"""

"""
##### GVEG (wskaźnik roślinności)
"""

def plot_histogram(ax, data, column, title, color):
    ax.hist(data[column], bins=20, color=color, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.set_xlim(min(min(CD[column]), min(CD_i_niepustynia[column]), min(niepustynia_przy_CD[column]),
                    min(CP[column]), min(CP_i_niepustynia[column]), min(niepustynia_przy_CP[column]),
                    min(GBD[column]), min(GBD_i_niepustynia[column]), min(niepustynia_przy_GBD[column])), 
                max(max(CD[column]), max(CD_i_niepustynia[column]), max(niepustynia_przy_CD[column]),
                    max(CP[column]), max(CP_i_niepustynia[column]), max(niepustynia_przy_CP[column]),
                    max(GBD[column]), max(GBD_i_niepustynia[column]), max(niepustynia_przy_GBD[column])))

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

plot_histogram(axes[0, 0], CD, 'GVEG', "CD_pustynia", "orange")
plot_histogram(axes[0, 1], CD_i_niepustynia, 'GVEG', "CD_pustynia/nie-pustynia", "yellow")
plot_histogram(axes[0, 2], niepustynia_przy_CD, 'GVEG', "CD_nie-pustynia", "green")

plot_histogram(axes[1, 0], CP, 'GVEG', "CP_pustynia", "orange")
plot_histogram(axes[1, 1], CP_i_niepustynia, 'GVEG', "CP_pustynia/nie-pustynia", "yellow")
plot_histogram(axes[1, 2], niepustynia_przy_CP, 'GVEG', "CP_nie-pustynia", "green")

plot_histogram(axes[2, 0], GBD, 'GVEG', "GBD_pustynia", "orange")
plot_histogram(axes[2, 1], GBD_i_niepustynia, 'GVEG', "GBD_pustynia/nie-pustynia", "yellow")
plot_histogram(axes[2, 2], niepustynia_przy_GBD, 'GVEG', "GBD_nie-pustynia", "green")

plt.tight_layout()
plt.show()


GVEG_graniczne = round(
    pd.Series([
        CD['GVEG'].quantile(0.75),
        niepustynia_przy_CD['GVEG'].quantile(0.25),
        CP['GVEG'].quantile(0.75),
        niepustynia_przy_CP['GVEG'].quantile(0.25),
        GBD['GVEG'].quantile(0.75),
        niepustynia_przy_GBD['GVEG'].quantile(0.25),
    ]).mean(skipna=True), 3
)


print("GVEG_graniczne:", GVEG_graniczne)


"""
##### Rainf (wskaźnik opadów deszczu)
"""

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

plot_histogram(axes[0, 0], CD, 'Rainf', "CD_pustynia", "orange")
plot_histogram(axes[0, 1], CD_i_niepustynia, 'Rainf', "CD_pustynia/nie-pustynia", "yellow")
plot_histogram(axes[0, 2], niepustynia_przy_CD, 'Rainf', "CD_nie-pustynia", "green")

plot_histogram(axes[1, 0], CP, 'Rainf', "CP_pustynia", "orange")
plot_histogram(axes[1, 1], CP_i_niepustynia, 'Rainf', "CP_pustynia/nie-pustynia", "yellow")
plot_histogram(axes[1, 2], niepustynia_przy_CP, 'Rainf', "CP_nie-pustynia", "green")

plot_histogram(axes[2, 0], GBD, 'Rainf', "GBD_pustynia", "orange")
plot_histogram(axes[2, 1], GBD_i_niepustynia, 'Rainf', "GBD_pustynia/nie-pustynia", "yellow")
plot_histogram(axes[2, 2], niepustynia_przy_GBD, 'Rainf', "GBD_nie-pustynia", "green")

plt.tight_layout()
plt.show()


Rainf_graniczne = round(
    pd.Series([
        CD['Rainf'].quantile(0.75),
        niepustynia_przy_CD['Rainf'].quantile(0.25),
        CP['Rainf'].quantile(0.75),
        niepustynia_przy_CP['Rainf'].quantile(0.25),
        GBD['Rainf'].quantile(0.75),
        niepustynia_przy_GBD['Rainf'].quantile(0.25),
    ]).mean(skipna=True)
)


print("Rainf_graniczne:", Rainf_graniczne)


"""
##### Evap (wskaźnik całkowitej ewapotranspiracji)
"""

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

plot_histogram(axes[0, 0], CD, 'Evap', "CD_pustynia", "orange")
plot_histogram(axes[0, 1], CD_i_niepustynia, 'Evap', "CD_pustynia/nie-pustynia", "yellow")
plot_histogram(axes[0, 2], niepustynia_przy_CD, 'Evap', "CD_nie-pustynia", "green")

plot_histogram(axes[1, 0], CP, 'Evap', "CP_pustynia", "orange")
plot_histogram(axes[1, 1], CP_i_niepustynia, 'Evap', "CP_pustynia/nie-pustynia", "yellow")
plot_histogram(axes[1, 2], niepustynia_przy_CP, 'Evap', "CP_nie-pustynia", "green")

plot_histogram(axes[2, 0], GBD, 'Evap', "GBD_pustynia", "orange")
plot_histogram(axes[2, 1], GBD_i_niepustynia, 'Evap', "GBD_pustynia/nie-pustynia", "yellow")
plot_histogram(axes[2, 2], niepustynia_przy_GBD, 'Evap', "GBD_nie-pustynia", "green")

plt.tight_layout()
plt.show()


Evap_graniczne = round(
    pd.Series([
        CD['Evap'].quantile(0.75),
        niepustynia_przy_CD['Evap'].quantile(0.25),
        CP['Evap'].quantile(0.75),
        niepustynia_przy_CP['Evap'].quantile(0.25),
        GBD['Evap'].quantile(0.75),
        niepustynia_przy_GBD['Evap'].quantile(0.25),
    ]).mean(skipna=True)
)


print("Evap_graniczne:", Evap_graniczne)


"""
##### AvgSurfT (wskaźnik średniej temperatury powierzchni ziemi)
"""

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

plot_histogram(axes[0, 0], CD, 'AvgSurfT', "CD_pustynia", "orange")
plot_histogram(axes[0, 1], CD_i_niepustynia, 'AvgSurfT', "CD_pustynia/nie-pustynia", "yellow")
plot_histogram(axes[0, 2], niepustynia_przy_CD, 'AvgSurfT', "CD_nie-pustynia", "green")

plot_histogram(axes[1, 0], CP, 'AvgSurfT', "CP_pustynia", "orange")
plot_histogram(axes[1, 1], CP_i_niepustynia, 'AvgSurfT', "CP_pustynia/nie-pustynia", "yellow")
plot_histogram(axes[1, 2], niepustynia_przy_CP, 'AvgSurfT', "CP_nie-pustynia", "green")

plot_histogram(axes[2, 0], GBD, 'AvgSurfT', "GBD_pustynia", "orange")
plot_histogram(axes[2, 1], GBD_i_niepustynia, 'AvgSurfT', "GBD_pustynia/nie-pustynia", "yellow")
plot_histogram(axes[2, 2], niepustynia_przy_GBD, 'AvgSurfT', "GBD_nie-pustynia", "green")

plt.tight_layout()
plt.show()


AvgSurfT_graniczne = round(
    pd.Series([
        CD['AvgSurfT'].quantile(0.25),
        niepustynia_przy_CD['AvgSurfT'].quantile(0.75),
        CP['AvgSurfT'].quantile(0.25),
        niepustynia_przy_CP['AvgSurfT'].quantile(0.75),
        GBD['AvgSurfT'].quantile(0.25),
        niepustynia_przy_GBD['AvgSurfT'].quantile(0.75),
    ]).mean(skipna=True)
)


print("AvgSurfT_graniczne:", AvgSurfT_graniczne)


"""
##### Albedo (wskaźnik albedo)
"""

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

plot_histogram(axes[0, 0], CD, 'Albedo', "CD_pustynia", "orange")
plot_histogram(axes[0, 1], CD_i_niepustynia, 'Albedo', "CD_pustynia/nie-pustynia", "yellow")
plot_histogram(axes[0, 2], niepustynia_przy_CD, 'Albedo', "CD_nie-pustynia", "green")

plot_histogram(axes[1, 0], CP, 'Albedo', "CP_pustynia", "orange")
plot_histogram(axes[1, 1], CP_i_niepustynia, 'Albedo', "CP_pustynia/nie-pustynia", "yellow")
plot_histogram(axes[1, 2], niepustynia_przy_CP, 'Albedo', "CP_nie-pustynia", "green")

plot_histogram(axes[2, 0], GBD, 'Albedo', "GBD_pustynia", "orange")
plot_histogram(axes[2, 1], GBD_i_niepustynia, 'Albedo', "GBD_pustynia/nie-pustynia", "yellow")
plot_histogram(axes[2, 2], niepustynia_przy_GBD, 'Albedo', "GBD_nie-pustynia", "green")
plt.tight_layout()
plt.show()


Albedo_graniczne = round(
    pd.Series([
        CD['Albedo'].quantile(0.25),
        niepustynia_przy_CD['Albedo'].quantile(0.75),
        CP['Albedo'].quantile(0.25),
        niepustynia_przy_CP['Albedo'].quantile(0.75),
        GBD['Albedo'].quantile(0.25),
        niepustynia_przy_GBD['Albedo'].quantile(0.75),
    ]).mean(skipna=True), 1
)


print("Albedo_graniczne:", Albedo_graniczne)


"""
##### SoilT_40_100cm (wskaźnik temperatury gleby w warstwie o głębokości od 40 do 100 cm)
"""

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

plot_histogram(axes[0, 0], CD, 'SoilT_40_100cm', "CD_pustynia", "orange")
plot_histogram(axes[0, 1], CD_i_niepustynia, 'SoilT_40_100cm', "CD_pustynia/nie-pustynia", "yellow")
plot_histogram(axes[0, 2], niepustynia_przy_CD, 'SoilT_40_100cm', "CD_nie-pustynia", "green")

plot_histogram(axes[1, 0], CP, 'SoilT_40_100cm', "CP_pustynia", "orange")
plot_histogram(axes[1, 1], CP_i_niepustynia, 'SoilT_40_100cm', "CP_pustynia/nie-pustynia", "yellow")
plot_histogram(axes[1, 2], niepustynia_przy_CP, 'SoilT_40_100cm', "CP_nie-pustynia", "green")

plot_histogram(axes[2, 0], GBD, 'SoilT_40_100cm', "GBD_pustynia", "orange")
plot_histogram(axes[2, 1], GBD_i_niepustynia, 'SoilT_40_100cm', "GBD_pustynia/nie-pustynia", "yellow")
plot_histogram(axes[2, 2], niepustynia_przy_GBD, 'SoilT_40_100cm', "GBD_nie-pustynia", "green")

plt.tight_layout()
plt.show()


SoilT_40_100cm_graniczne = round(
    pd.Series([
        CD['SoilT_40_100cm'].quantile(0.25),
        niepustynia_przy_CD['SoilT_40_100cm'].quantile(0.75),
        CP['SoilT_40_100cm'].quantile(0.25),
        niepustynia_przy_CP['SoilT_40_100cm'].quantile(0.75),
        GBD['SoilT_40_100cm'].quantile(0.25),
        niepustynia_przy_GBD['SoilT_40_100cm'].quantile(0.75),
    ]).mean(skipna=True)
)


print("SoilT_40_100cm_graniczne:", SoilT_40_100cm_graniczne)


"""
# Klasyfikacja: pustynia / nie-pustynia
"""

NASA = NASA[['lon', 'lat', 'Rainf', 'Evap', 'AvgSurfT', 'Albedo', 'SoilT_40_100cm', 'GVEG']].dropna()
NASA['klasyfikacja'] = np.nan


def classify(row: pd.DataFrame):
    conditions = [
        row['Rainf'] <= Rainf_graniczne,
        row['Evap'] <= Evap_graniczne,
        row['GVEG'] <= GVEG_graniczne,
        row['AvgSurfT'] >= AvgSurfT_graniczne,
        row['Albedo'] >= Albedo_graniczne,
        row['SoilT_40_100cm'] >= SoilT_40_100cm_graniczne
    ]
    if np.nansum(conditions) >= 4:
        return "pustynia"
    else:
        return "nie-pustynia"


NASA['klasyfikacja'] = NASA.apply(classify, axis=1)


NASA.head(10)


pustynia_percentage = (NASA['klasyfikacja'] == "pustynia").sum() / len(NASA)


print("pustynia_percentage:", pustynia_percentage)