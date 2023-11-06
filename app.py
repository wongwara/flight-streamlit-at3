import streamlit as st
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, 'flight-prediction/src/')
sys.path.insert(0, 'flight-prediction/models/')
from models.predict_model import print_regressor_score

def main():
    st.title("Data Product with Machine Learning")

@st.cache
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
    X = df[['totalTravelDistance', 'isNonStop', 'isBasicEconomy', 'startingAirport', 'destinationAirport', 'segmentsCabinCode', 'flightDate_day', 'flightDate_month', 'flightDate_year',
             'DepartTime_hour', 'DepartTime_minute', 'DepartTime_second']]
    y = df['totalFare']
    return X, y

@st.cache
def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@st.cache
def model_building(X_train, y_train):
    rf = RandomForestRegressor(max_features=4, n_estimators=100)
    rf.fit(X_train, y_train)
    y_train_pred = rf.predict(X_train)
    return y_train_pred

@st.cache
def model_performance(y_train, y_train_pred, y_test, y_test_pred):
    y_test_pred = rf.predict(X_test)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    return train_r2, test_r2, train_mse, test_mse

if __name__ == '__main__':
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    y_train_pred = model_building(X_train, y_train)
    train_r2, test_r2, train_mse, test_mse = model_performance(y_train, y_train_pred, y_test, y_test_pred)

    st.title('🤖 Scikit-learn - A fare prediction')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header('Actual y')
        st.write(y_train)

    with col2:
        st.header('Predicted y')
        st.write(y_train_pred)

    with col3:
        st.header('Performance')
        st.metric(label="**Train $R^2$**", value=round(train_r2, 3), delta="0")
        st.metric(label="**Test $R^2$**", value=round(test_r2, 3), delta="0")
