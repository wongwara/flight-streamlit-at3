import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import numpy as np

def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/wongwara/fare-prediction/main/data/sample_itineraries.csv")
    
    return df

df = load_data()

def show_explore_page():
    st.title("💰 Fare Prediction")
  
    st.write(
        """ 
        The task is to build a data product that will help users in the USA to better estimate their local travel airfare. Users will be able to provide details of their trip and the app will predict the expected flight fare.
        """
    ) 
    st.write(
        """
             Therefore, the objective of this project would be to develop a machine learning model that accepts airport name and flightdate return the predict total fare.
             """
            )