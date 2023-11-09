from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import warnings
import tensorflow as tf
import tensorflow_decision_forests as tfdf
warnings.filterwarnings("ignore")
import sys
sys.path.append('../../src')
from data.make_dataset import load_sets

df = pd.read_csv('../path_to_new_data')

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


########## Train models #############

## 1. KNN model
from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=9, weights='uniform',p=1)
knn_model.fit(X_train, y_train)

## 2. XGB model
import xgboost as xgb
bestparam_xgb_model = xgb.XGBRegressor(
    n_estimators=300,     # Best number of boosting rounds
    max_depth=5,          # Best maximum depth of trees
    learning_rate=0.2     # Best learning rate
)
bestparam_xgb_model.fit(X_train, y_train)

## 3. TF-DF model
# Convert data to tf.data.Datasets
batch_size = 100
# Assuming X_train, y_train, X_val, y_val, X_test, and y_test are NumPy arrays or tensors
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.batch(batch_size)
validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
validation_dataset = validation_dataset.batch(batch_size)

# Define the TensorFlow Decision Forest model
model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
model.compile(metrics=["mean_squared_error"])
model.fit(train_dataset, epochs=1, validation_data=validation_dataset)

## 4. Keras
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
import keras_tuner

def build_model_tune(hp):
  model_tune = Sequential()
  model_tune.add(Dense(hp.Choice('units', [8, 16, 32]), input_shape=(12,), activation='relu'))
  model_tune.add(Dense(1, activation='linear'))

  model_tune.compile(loss='mean_squared_error', optimizer='adam')
  return model_tune

tuner = keras_tuner.RandomSearch(
    build_model_tune,
    objective='val_loss',
    max_trials=5)
tuner.search(X_train, y_train, epochs=5, validation_data=(X_val, y_val))
best_model = tuner.get_best_models()[0]