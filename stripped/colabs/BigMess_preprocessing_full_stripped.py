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

from big_mess.loaders import default_loader, save_to_csv

spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()

nasa = default_loader(spark)

save_to_csv(nasa, '/content/drive/MyDrive/BigMess/NASA/NASA_preprocessed.csv')


