import streamlit as st
import numpy as np
import pandas as pd 

def main():
    st.title("Data Product with Machine Learning")

def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/wongwara/flight-streamlit-at3/main/data/sample_itineraries.csv?token=GHSAT0AAAAAACETSXPAXBU2LOS7R6N4FXIOZKIO7MA')
    X = df[['totalTravelDistance', 'isNonStop', 'isBasicEconomy', 'startingAirport', 'destinationAirport', 'segmentsCabinCode', 'flightDate_day', 'flightDate_month', 'flightDate_year',
             'DepartTime_hour', 'DepartTime_minute', 'DepartTime_second']]
    y = df['totalFare']
    return X, y


if __name__ == '__main__':
    X, y = load_data()
    
