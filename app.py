import streamlit as st
import sys
import numpy as np
import pandas as pd 
import xgboost 

from models.predict_model import print_regressor_scores
sys.path.insert(0, 'flight-prediction/src/')
sys.path.insert(0, 'flight-prediction/models/')

def main():
    st.title("Data Product with Machine Learning")
    
