# 
import pandas as pd
import numpy as np

from states.state import OverallState

def column_statistics(dataframe):
    """
    Get statistics for each column in the dataframe
    Parameters
    ----------
    dataframe : pd.DataFrame
    The dataframe for which to calculate statistics
    Returns
    -------
    pd.DataFrame
    A dataframe containing statistics for each column in the input dataframe
    """
    stats = {}
    for column, series in dataframe.items():
        stats[column] = {}
        valid_values = series.dropna()
        if valid_values.empty or len(valid_values.index) <= 1:
            stats[column]["Maximum Gap Between Records"] = None
            stats[column]["Average Gap Between Records"] = None
        else:
            time_diffs = np.diff(valid_values.index)
            stats[column]["Maximum Gap Between Records"] = time_diffs.max()
            stats[column]["Average Gap Between Records"] =  time_diffs.mean()
            
        if series.dtype in ['float64', 'int64', 'int32', 'float32']:
            stats[column]['Mean'] = valid_values.mean()
            stats[column]['Median'] = valid_values.median()
            stats[column]['IQR'] = valid_values.quantile(0.75) - valid_values.quantile(0.25)
            stats[column]['Max - Min'] = valid_values.max() - valid_values.min()
            
    df = pd.DataFrame(stats)
    df.loc["Missing Time Before First Valid Value"] = dataframe.apply(lambda col: (col.first_valid_index() - dataframe.index[0]), axis=0)
    df.loc["Missing Time After Last Valid Value"] = dataframe.apply(lambda col: (dataframe.index[-1] - col.last_valid_index()), axis=0)
    df.loc["Data Type of Column"] = dataframe.dtypes
    df.loc["Percentage of Unique Values"] = (dataframe.nunique() / dataframe.count()) * 100
    df.loc["percenatge of Missing Value "] = 100 -dataframe.count()/len(dataframe)*100
    df.loc["Total Number of Data Points"] = len(dataframe)
    return df

def data_statistics_analysis(state: OverallState):
    """
    Analyze and compute statistics for the dataset.
    
    Parameters:
        state (OverallState): A dictionary-like object containing the dataset and its structure metadata.

    Returns:
        dict: A dictionary with computed statistics for each column in the dataset.
    """
    dataframe = state["dataframe"]

    if state["structure"]["type"].lower() == "time series":
        time_series_column = state["structure"].get("columns")
        if time_series_column not in dataframe.columns:
            raise ValueError(f"Column '{time_series_column}' specified for time series index not found in the dataframe.")
        dataframe = dataframe.set_index(time_series_column)
    
    try:
        statistics = column_statistics(dataframe)
    except Exception as e:
        raise RuntimeError(f"Error calculating statistics: {e}")
    
    return {"statistics": statistics}


