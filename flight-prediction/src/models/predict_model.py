import tensorflow as tf
import os
from tensorflow import keras
import joblib

def print_regressor_scores(y_preds, y_actuals, set_name=None):
    """Print the RMSE and MAE for the provided data

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae
    from sklearn import metrics

    print(f"MSE {set_name}: {mse(y_actuals, y_preds)}")
    print(f"MAE {set_name}: {mae(y_actuals, y_preds)}")
    print(f"R2_score {set_name}: {round(metrics.r2_score(y_actuals, y_preds),6)}")

def assess_regressor_set(model, features, target, set_name=''):
    """Save the predictions from a trained model on a given set and print its RMSE and MAE scores

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Trained Sklearn model with set hyperparameters
    features : Numpy Array
        Features
    target : Numpy Array
        Target variable
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    preds = model.predict(features)
    print_regressor_scores(y_preds=preds, y_actuals=target, set_name=set_name)

def fit_assess_regressor(model, X_train, y_train, X_val, y_val):
    """Train a regressor model, print its RMSE and MAE scores on the training and validation set and return the trained model

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Instantiated Sklearn model with set hyperparameters
    X_train : Numpy Array
        Features for the training set
    y_train : Numpy Array
        Target for the training set
    X_train : Numpy Array
        Features for the validation set
    y_train : Numpy Array
        Target for the validation set

    Returns
    sklearn.base.BaseEstimator
        Trained model
    -------
    """
    model.fit(X_train, y_train)
    assess_regressor_set(model, X_train, y_train, set_name='Training')
    assess_regressor_set(model, X_val, y_val, set_name='Validation')
    return model

##########  Load best models  ###############
  
# Get the absolute path to the models directory
models_dir = os.path.abspath('../../models')
# XGB
xgb_model_path = os.path.join(models_dir, 'xgb_bestparam.joblib')
xgb_model = joblib.load(xgb_model_path)
# KNN
knn_regressor_loaded_path = os.path.join(models_dir, 'knn_fit.joblib')
knn_regressor_loaded = joblib.load(knn_regressor_loaded_path)
# TF-DF
tfdf_model_path = os.path.join(models_dir, 'tfdf_model')
tfdf_model = tf.keras.models.load_model(tfdf_model_path)
# KERAS
keras_model_path = os.path.join(models_dir, 'keras.keras')
keras_model = keras.models.load_model(keras_model_path)


############## Prediction #################

total_fare_knn = knn_regressor_loaded.predict(X)
total_fare_xg = xgb_model.predict(X)
total_fare_tfdf = tfdf_model.predict(X)
total_fare_keras = keras_model.predict(X)


# Print prediction
print(f'The total fare for your trip with KNN model: ${total_fare_tfdf[0]:.2f}')
print(f'The total fare for your trip with XGB model: ${total_fare_tfdf[0]:.2f}')
print(f'The total fare for your trip with TF-DF model: ${total_fare_tfdf[0][0]:.2f}')
print(f'The total fare for your trip with Keras model: ${total_fare_keras[0][0]:.2f}')