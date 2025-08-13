import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d


def train_blender(historical_df):
    """
    Train a Linear Regression model to blend forecasts from multiple models.

    Parameters
    ----------
    historical_df : pd.DataFrame
        DataFrame with columns:
        ['LSTM', 'GARCH', 'XGB', 'Informer', 'Autoformer', 'Actual']

    Returns
    -------
    model : LinearRegression
        Trained Linear Regression model
    weights : np.ndarray
        Normalized weights of each model
    """
    X = historical_df[['LSTM', 'GARCH', 'XGB', 'Informer', 'Autoformer']].values
    y = historical_df['Actual'].values
    model = LinearRegression()
    model.fit(X, y)
    weights = model.coef_ / np.sum(np.abs(model.coef_))
    return model, weights


def smooth_predictions(preds, method="linear"):
    """
    Smooth predictions using interpolation.

    Parameters
    ----------
    preds : list or np.ndarray
        Sequence of predicted prices
    method : str
        Interpolation method (linear, quadratic, cubic, etc.)

    Returns
    -------
    np.ndarray
        Smoothed prediction series
    """
    x = np.arange(len(preds))
    f = interp1d(x, preds, kind=method, fill_value="extrapolate")
    x_new = np.linspace(0, len(preds) - 1, num=50)
    return f(x_new)


def get_final_prediction(predictions, weights):
    """
    Compute the final blended prediction from multiple model forecasts.

    Parameters
    ----------
    predictions : list or np.ndarray
        Forecasts from the different models
    weights : np.ndarray
        Corresponding model weights

    Returns
    -------
    float
        Final blended prediction
    """
    predictions = np.array(predictions)
    return np.dot(weights, predictions)


def prepare_historical_df(pred_dict, actual_series):
    """
    Prepare historical dataframe for training the blender.

    Parameters
    ----------
    pred_dict : dict
        Dictionary containing historical predictions from each model
    actual_series : pd.Series
        Actual observed prices

    Returns
    -------
    pd.DataFrame
        DataFrame formatted for train_blender
    """
    df = pd.DataFrame(pred_dict)
    df['Actual'] = actual_series
    return df
