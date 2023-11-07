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
    import matplotlib.pyplot as plt
    # Group data by month and aggregate total fare
    df['flightDate'] = pd.to_datetime(df['flightDate'])  # Ensure 'flightDate' is in datetime format
    df['Month'] = df['flightDate'].dt.to_period('M')  # Extract the month from 'flightDate'
    grouped_df = df.groupby('Month')['totalFare'].sum().reset_index()

    # Create a line plot
    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the figure size as needed
    ax.plot(grouped_df['Month'], grouped_df['totalFare'], label='Total Fare', color='b', marker='o')
    ax.set_title('Total Fare Over Time (Grouped by Month)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Fare')
    ax.grid(True)
    ax.legend()
    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability

    # Display the plot in Streamlit
    st.pyplot(fig)
