
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score


def regressor_metrics(model, y_hat, y_true):
    """
    args:
        model (str): name of regression model
        y_hat (array): predicted values
        y_true (array): true values
        
    returns:
        rmse (float): root mean squared error
        mae (float): mean absolute error
        r2 (float): r2 score
        evs (float): explained variance score
    """
    
    regres_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_hat)) / 2,
        'MAE': mean_absolute_error(y_true, y_hat),
        'R2': r2_score(y_true, y_hat),
        'EVS': explained_variance_score(y_true, y_hat)
    }
    
    regr_df = pd.DataFrame.from_dict(regres_metrics, orient='index')
    regr_df.columns = [model]
    
    return regr_df

def RMSE(y_hat, y_true):
    return np.sqrt(mean_squared_error(y_true, y_hat)) / 2




def plot_feature_weights(model, weights, names, types = 'c'):
    """
    args:
        model (str): name of regression model
        weights (array): feature weights
        names (array): feature names
        type (str): type of plot
    
    returns:
        plot of feature weights/importances
    
    """
    plt.style.use('fivethirtyeight')
    
    (px, py) = (9, 7) if types == 'c' else (9, 8)
    W = pd.DataFrame({'weights': weights}, index=names)
    W.sort_values('weights', ascending=True).plot(kind='barh', figsize=(px, py), color='r', title=model)
    plt.ylabel('Features')
    label = 'feature coefficients' if types == 'c' else 'feature importances'
    plt.xlabel(model + ' ' + label)
    plt.show()    
    
    
    
def plot_residuals(model, y_train_hat, y_train_true, y_val_hat, y_val_true):
    """
    args:
        model (str): name of regression model
        y_train_hat (array): predicted values of training set
        y_train_true (array): true values of training set
        y_val_hat (array): predicted values of validation set
        y_val_true (array): true values of validation set
        
    returns:
        plot of residuals
    """
    
    residuals_train = y_train_hat - y_train_true
    residuals_val = y_val_hat - y_val_true
    
    plt.style.use('ggplot')
    
    plt.figure(figsize=(10,5))
    plt.scatter(y_train_hat, residuals_train, color='b', marker='o', label= 'Training data')
    plt.scatter(y_val_hat, residuals_val, color='y', marker='s', label= 'Validation data')
    plt.title(model + ' Residuals plot')
    plt.ylabel('Residuals')
    plt.xlabel('Predicted Values')
    plt.axhline(y=0, xmin=-60, xmax=500, color='r', lw=2)
    plt.legend(loc='upper left')
    plt.show()
    
    
    