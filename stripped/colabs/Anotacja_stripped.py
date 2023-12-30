import random
import pandas as pd
import os
from google.colab import drive
drive.mount("/content/drive")

NASA = pd.read_csv('/content/drive/MyDrive/sampled_NASA_200k.csv')

selected_columns = ["lon", "lat"]
NASA_coordinates = NASA[selected_columns]

NASA_coordinates = NASA_coordinates.drop_duplicates(subset=["lat", "lon"])

random.seed(1)
random_sample = random.sample(range(0, len(NASA_coordinates) - 1), 100)

NASA_sample = NASA_coordinates.iloc[random_sample, :].copy()

NASA_sample

NASA_sample.to_csv('/content/drive/MyDrive/NASA_sample.csv', index=False, sep=';')

NASA_sample_an = pd.read_csv('/content/drive/MyDrive/BigMess/NASA/NASA_sample_an.csv', sep=';')

NASA_sample_an

NASA_sample_an['pustynia'].sum()

NASA_sample_an['step'].sum()


