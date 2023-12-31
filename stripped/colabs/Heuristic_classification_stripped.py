from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/PiotrMaciejKowalski/BigData2024Project.git
%cd BigData2024Project
!git checkout refactoring-sprint2
%cd ..

%cd BigData2024Project
!git pull
%cd ..

%cd BigData2024Project
!git status
%cd ..

!chmod 755 /content/BigData2024Project/src/setup.sh
!/content/BigData2024Project/src/setup.sh

import sys
sys.path.append('/content/BigData2024Project/src')

from start_spark import initialize_spark
initialize_spark()

from pyspark.sql import SparkSession
from big_mess.heuristic_classifier import heuristic_classify
from big_mess.loaders import preprocessed_loader, save_to_csv

spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()

data=preprocessed_loader(spark,'/content/drive/MyDrive/BigMess/NASA/NASA_month_preprocessed.csv')

data.show(5)

result = heuristic_classify(data)
result.show(20)

save_to_csv(result.drop("Rainf_condition","Evap_condition","GVEG_condition","AvgSurfT_condition","Albedo_condition","conditions_fullfiled_sum"), '/content/drive/MyDrive/BigMess/NASA/NASA_heuristic.csv')

2+2
