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

!pip install imblearn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler

from pyspark.sql import SparkSession

from big_mess.loaders import preprocessed_loader


spark = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()

data=preprocessed_loader(spark,'/content/drive/MyDrive/BigMess/NASA/NASA_anotated_preprocessed.csv')

data.show()

df = (
    data
    .drop('SoilT_40_100cm') #remove after fixing null values there
    .toPandas()
)
df.head()

df['pustynia'].hist()

target_column = 'pustynia'
remove_columns = ['lon', 'lat', 'Year', 'Month']
training_columns = [column for column in df.columns if column != target_column and column not in remove_columns]

X = df[training_columns]
y = df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LogisticRegression()

oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)


# Fit the model on the training data
model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the metrics
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)
