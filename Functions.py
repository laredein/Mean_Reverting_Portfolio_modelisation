# Libraries required
import numpy as np
import pandas as pd
import scipy


def spread_From_Excel(filename, path = ""):
    # Get the series from Excel file, delete NAN cells from the end and
    # put it into numpy format for more comfortable usage
    SpreadSeries = pd.read_excel(path + filename)["Spread"]
    SpreadSeries.dropna(axis=0, inplace=True)
    SpreadSeries = np.array(SpreadSeries)
    return SpreadSeries

def series_Moments(Series):
    # returns 4 moments in order starting from the first
    mean = np.mean(Series)
    variance = np.var(Series)
    exkurtosis = scipy.stats.kurtosis(Series)
    skewness = scipy.stats.skew(Series)
    return mean, variance, exkurtosis, skewness

def series_Increment_Moments(Series):
    increments = np.diff(Series)
    # I am not sure if kurtosis and skewness makes any sense in this contest
    return series_Moments(increments)[0], series_Moments(increments)[1]




