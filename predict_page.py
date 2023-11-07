import streamlit as st
import pandas as pd
import re
from datetime import datetime, time
from prediction import load_knn_model
import tensorflow as tf
import numpy as np
import joblib
import os
from tensorflow import keras

# Get the absolute path to the models directory
models_dir = os.path.abspath('models')

# Load the XGB model with the full path
xgb_model_path = os.path.join(models_dir, 'xgb_bestparam.joblib')
xgb_model = joblib.load(xgb_model_path)

knn_regressor_loaded = load_knn_model()

tfdf_model_path = os.path.join(models_dir, 'tfdf_model')
tfdf_model = tf.keras.models.load_model(tfdf_model_path)

keras_model_path = os.path.join(models_dir, 'keras_model.keras')
keras_model = keras.models.load_model(keras_model_path)

def show_predict_page():
    st.title(" ✈️ Fare Prediction")
    st.write(""" This project is for the user and students to search the total fare from related information""")
    st.subheader("We need some information to predict the total Fare for your trip")

    # Input flight date with a calendar widget
    flightDate = st.date_input("Flight Date", value=datetime.today(), format="YYYY-MM-DD")
    # Starting Airport
    airport_dict = {
        'OAK - Oakland International Airport':15,
        'DEN - Denver International Airport':3,
        'LGA - LaGuardia Airport':10,
        'LAX - Los Angeles International Airport':9,
        'ATL - Hartsfield-Jackson Atlanta International Airport':0,
        'CLT - Charlotte Douglas International Airport':2,
        'PHL - Philadelphia International Airport':13,
        'DTW - Detroit Metropolitan Wayne County Airport':5,
        'IAD - Washington Dulles International Airport':7,
        'JFK - John F. Kennedy International Airport':8,
        'DFW - Dallas/Fort Worth International Airport':4,
        'BOS - Logan International Airport':1,
        'EWR - Newark Liberty International Airport':6,
        'SFO - San Francisco International Airport':14,
        'ORD - O Hare International Airport':12,
        'MIA - Miami International Airport':11
            }
    airport_options = list(airport_dict.keys())
    starting_airport = st.selectbox("Starting Airport", airport_options)
    startingAirport = airport_dict[starting_airport]
    
    destination_airport = st.selectbox("Destination Airport", airport_options)
    destinationAirport = airport_dict[destination_airport]
    if starting_airport == destination_airport:
        st.error("Starting and destination airports cannot be the same.")
    else:
    # Convert the selected airports to their corresponding codes
        startingAirport = airport_dict[starting_airport]
        destinationAirport = airport_dict[destination_airport]
        
    # Set the default time to 10:30 AM
    default_time = time(10, 30)
    # Input departure time 
    departure_time = st.time_input('Departure time', value=default_time)
    
    #Input cabin code
    cabin_dict = {'coach - coach':0,
                  'coach - coach - coach':1,
                  'coach':2,
                  'coach - coach - premium coach':3,
                  'first - first - first':4,
                  'coach - coach - coach - coach':5,
                  'coach - first - first':6,
                  'coach - premium coach - premium coach':7,
                  'coach - first - coach':8,
                  'first - first - coach':9,
                  'coach - business - business':10,
                  'business - coach':11,
                  'coach - premium coach - coach':12,
                  'first - first':13,
                  'first - coach':14,
                  'coach - coach - first':15,
                  'coach - business':16,
                  'business - coach - coach':17,
                  'business - business':18,
                  'first - coach - coach':19,
                  'coach - business - coach':20,
                  'first - coach - first':21,
                  'coach - coach - business':22,
                  'coach - premium coach':23,
                  'premium coach':24,
                  'business - business - coach':25,
                  'coach - first':26,
                  'business - coach - business':27,
                  'first':28,
                  'premium coach - premium coach':29,
                  'coach - coach - coach - premium coach':30,
                  'premium coach - coach':31,
                  'premium coach - coach - coach':32,
                  'business':33,
                  'first - coach - business':34,
                  'coach - coach - coach - first':35,
                  'premium coach - premium coach - premium coach':36,
                  'premium coach - premium coach - coach':37,
                  'first - business':38,
                  'first - first - coach - coach':39,
                  'first - coach - coach - coach':40,
                  'premium coach - coach - coach - coach':41,
                  'premium coach - first':42,
                  'coach - business - first':43,
                  'business - first':44,
                  'business - first - first':45,
                  'premium coach - business - coach':46,
                  'coach - coach - first - coach':47,
                  'coach - coach - premium coach - premium coach':48,
                  'coach - coach - first - first':49,
                  'coach - coach - premium coach - coach':50,
                  'coach - coach - business - coach':51
                 }
    cabin_options = list(cabin_dict.keys())
    cabin = st.selectbox("cabin code", cabin_options)
    segmentsCabinCode = cabin_dict[cabin]
    # Set isNonStop based on segmentsCabinCode and specified conditions
    if segmentsCabinCode in [28, 33, 2, 24]:
        isNonStop = True
    else:
        isNonStop = False
    isNonStop = int(isNonStop)
    ok = st.button("Calculate total fare for your trip")
    if ok:
        X = pd.DataFrame({
        'flightDate':[flightDate],
        'startingAirport':[startingAirport],
        'destinationAirport':[destinationAirport],
        'departureTime': [departure_time], 
        'segmentsCabinCode':[segmentsCabinCode],
        'isNonStop': [isNonStop], 
        'isBasicEconomy': [0],  # Set default value to False
        'totalTravelDistance': [1569.618] #Mean total travel distance
        })
    
        # Transform date column: flightDate
        X['flightDate'] = pd.to_datetime(X['flightDate'])
        X['flightDate_day'] = X['flightDate'].dt.day
        X['flightDate_month'] = X['flightDate'].dt.month
        X['flightDate_year'] = X['flightDate'].dt.year
        
        hour = departure_time.hour
        minute = departure_time.minute
        X['DepartTime_hour'] = hour
        X['DepartTime_minute'] = minute
        X['DepartTime_second'] = 0  # Since seconds are always 00

        # Drop date columns
        X = X.drop(columns=['flightDate','departureTime'])
        X = X[['totalTravelDistance', 'isNonStop', 'isBasicEconomy', 'startingAirport', 'destinationAirport', 'segmentsCabinCode','flightDate_day', 'flightDate_month', 'flightDate_year',
                         'DepartTime_hour', 'DepartTime_minute', 'DepartTime_second']]

        total_fare_knn = knn_regressor_loaded.predict(X)
        total_fare_knn = np.round(total_fare_knn, 2)  # Round the value to two digits
        total_fare_str_knn = str(total_fare_knn[0])  # Convert to string
        st.write(f"The total fare for your trip with KNN regressor {total_fare_str_knn}$")
        
        total_fare_xg = xgb_model.predict(X)
        total_fare_xg = np.round(total_fare_xg, 2)  # Round the value to two digits
        total_fare_str_xg = str(total_fare_xg[0])  # Convert to string
        st.write(f"The total fare for your trip with XGBoost Regressor {total_fare_str_xg}$")

        total_fare_tfdf = tfdf_model.predict(X)
        total_fare_tfdf = np.round(total_fare_tfdf, 2)  # Round the value to two digits
        total_fare_str_tfdf = str(total_fare_tfdf[0][0])  # Convert to string
        st.write(f"The total fare for your trip with tensorflow keras {total_fare_str_tfdf}$")
        
        total_fare_keras = keras_model.predict(X)
        total_fare_keras = np.round(total_fare_keras, 2)  # Round the value to two digits
        total_fare_str_keras = str(total_fare_keras[0][0])  # Convert to string
        st.write(f"The total fare for your trip with keras {total_fare_str_keras}$")
