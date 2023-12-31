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

import pandas as pd
from pyspark.sql import SparkSession

from big_mess.loaders import default_loader, load_single_month, load_anotated, save_to_csv

spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()

%%time
nasa_full = default_loader(spark)
nasa_full.count()

save_to_csv(nasa_full, '/content/drive/MyDrive/BigMess/NASA/NASA_full_preprocessed.csv')

%%time
nasa_single_month = load_single_month(spark)
nasa_single_month.count()

save_to_csv(nasa_single_month, '/content/drive/MyDrive/BigMess/NASA/NASA_month_preprocessed.csv')

%%time
nasa_anotated = load_anotated(spark)
nasa_anotated.count()

save_to_csv(nasa_anotated, '/content/drive/MyDrive/BigMess/NASA/NASA_anotated_preprocessed.csv')

2+2
