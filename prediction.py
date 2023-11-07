import numpy as np
import pandas as pd
import warnings
import tensorflow as tf
import tensorflow_decision_forests as tfdf
warnings.filterwarnings("ignore")
df = pd.read_csv("https://raw.githubusercontent.com/wongwara/fare-prediction/main/data/sample_itineraries.csv")

#transform date column: flightDate
df['flightDate'] = pd.to_datetime(df['flightDate'])
df['flightDate_day'] = df['flightDate'].dt.day
df['flightDate_month'] = df['flightDate'].dt.month
df['flightDate_year'] = df['flightDate'].dt.year

#drop date cols
df = df.drop(columns=['searchDate', 'flightDate'])

df['DepartTime'] = df['segmentsDepartureTimeEpochSeconds'].apply(lambda x: x[1:-1].split(',')[0] if isinstance(x, str) else x)
df['DepartTime'] = pd.to_datetime(df['DepartTime'], unit='s')

# Extract and create new columns for hours, minutes, and seconds
df['DepartTime_hour'] = df['DepartTime'].dt.hour
df['DepartTime_minute'] = df['DepartTime'].dt.minute
df['DepartTime_second'] = df['DepartTime'].dt.second

df = df[['totalTravelDistance', 'isNonStop', 'isBasicEconomy', 'startingAirport', 'destinationAirport', 'segmentsCabinCode','flightDate_day', 'flightDate_month', 'flightDate_year',
                         'DepartTime_hour', 'DepartTime_minute', 'DepartTime_second','totalFare']]

cols = df.columns.to_list()
num_cols = df.select_dtypes(np.number).columns.to_list()
cat_cols = list(set(cols) - set(num_cols))
df['totalTravelDistance']= df['totalTravelDistance'].fillna(df['totalTravelDistance'].mean())

## Lable Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df[cat_cols] = df[cat_cols].apply(le.fit_transform)
#scale numeric column
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])

def pop_target(df, target_col):
    df_copy = df.copy()
    target = df_copy.pop(target_col)

    return df_copy, target

features, target = pop_target(df, 'totalFare')

def split_sets_random(features, target, test_ratio=0.2):
    from sklearn.model_selection import train_test_split
    val_ratio = test_ratio / (1 - test_ratio)
    X_data, X_test, y_data, y_test = train_test_split(features, target, test_size=test_ratio, random_state=8)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=val_ratio, random_state=8)
    return X_train, y_train, X_val, y_val, X_test, y_test
  
X_train, y_train, X_val, y_val, X_test, y_test = split_sets_random(features, target, test_ratio=0.2)
import joblib
import os

# # Get the absolute path to the models directory
# models_dir = os.path.abspath('models')

# # Load the KNN model with the full path
# # knn_model_path = os.path.join(models_dir, 'knn.joblib')
# knn_model = joblib.load(knn_model_path)
from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=3, weights='uniform',p=1)
knn_model.fit(X_train, y_train)

# Model evaluation for training set
y_train_preds_knn = knn_model.predict(X_train)
y_test_preds_knn = knn_model.predict(X_test)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

# Model evaluation for training set
y_train_preds_lr = lr.predict(X_train)
y_test_preds_lr = lr.predict(X_test)

from tensorflow_decision_forests import keras

# Define your model. Example: "RandomForestModel" for a random forest.
keras_model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)

# Convert data to tf.data.Datasets
batch_size = 100

# We have X_train, y_train, X_val, y_val, X_test, and y_test as NumPy arrays or tensors
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.batch(batch_size)

validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
validation_dataset = validation_dataset.batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(batch_size)

# Build the model
keras_model.compile(metrics=["mean_squared_error"])
keras_model.fit(train_dataset, epochs=1, validation_data=validation_dataset)
keras_model.predict(test_dataset)
import pickle
# Save the linear regression model
data = {"model": lr}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)

# Save the KNN model
knn = {"knn_model": knn_model}
with open('saved_knn.pkl', 'wb') as file:
    pickle.dump(knn, file)

keras ={"keras_model": keras_model}
with open('saved_keras.pkl', 'wb') as file:
    pickle.dump(keras, file)
  
# Load the linear regression model
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data["model"]

# Load the KNN model
def load_knn_model():
    with open('saved_knn.pkl', 'rb') as file:
        knn = pickle.load(file)
    return knn["knn_model"]
  
# Load the Keras model
def load_keras_model():
    with open('saved_keras.pkl', 'rb') as file:
        keras = pickle.load(file)
    return keras["keras_model"]

regressor_loaded = load_model()
knn_regressor_loaded = load_knn_model()
keras_regressor_loaded = load_keras_model()
