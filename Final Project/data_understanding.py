# IMPORTING PACKAGES

# FOR FILE
import zipfile

# DATA PREPROCESSING
import pandas as pd
import datetime 
import numpy as np
import re

# VISUALIZATION
import folium
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# MACHINE LEARNING PACKAGES
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def load_and_preprocess_data():
    ## LOADING DATA

    ## Unzipfile
    zip_file_sample = zipfile.ZipFile("./pkdd-15-predict-taxi-service-trajectory-i/sampleSubmission.csv.zip")
    zip_file_train = zipfile.ZipFile("./pkdd-15-predict-taxi-service-trajectory-i/train.csv.zip")
    zip_file_test = zipfile.ZipFile("./pkdd-15-predict-taxi-service-trajectory-i/test.csv.zip")
    zip_file_GPSlocation = zipfile.ZipFile("./pkdd-15-predict-taxi-service-trajectory-i/metaData_taxistandsID_name_GPSlocation.csv.zip")

    # Converting Files To Pandas Dataframe
    sample = pd.read_csv(zip_file_sample.open('sampleSubmission.csv'))
    train = pd.read_csv(zip_file_train.open("train.csv"))
    test = pd.read_csv(zip_file_test.open("test.csv"))
    location = pd.read_csv(zip_file_GPSlocation.open("metaData_taxistandsID_name_GPSlocation.csv"))

    # DROPPING "DAY_TYPE" COLUMN
    if "DAY_TYPE" in train.columns:
        train = train.drop("DAY_TYPE", axis=1)
    if "DAY_TYPE" in test.columns:
        test = test.drop("DAY_TYPE", axis=1)
    
    return train, test, sample, location

if __name__ == "__main__":
    train, test, sample, location = load_and_preprocess_data()

    print("sample\n", sample, "\n")
    print("train\n", train, "\n")
    print("test\n", test, "\n")
    print("location\n", location, "\n")

    print(train.info())
    print(test.info())

    Pcent_missing_train = train.isnull().sum() * 100 / len(train)
    print("Pcent_missing_train\n", Pcent_missing_train, "\n")
    Pcent_missing_test = test.isnull().sum() * 100 / len(test)
    print("Pcent_missing_test\n", Pcent_missing_test, "\n")

    print("train after dropping DAY_TYPE\n", train, "\n")
    print("test after dropping DAY_TYPE\n", test, "\n")

    # CONDUCTING SUMMARY STATISTICS 
    print("train summary statistics\n", train.describe(), "\n")
    print("test summary statistics\n", test.describe(), "\n")

    # INSIGHTS INTO TRAIN DATATYPES
    print("train datatypes\n", train.dtypes, "\n")
    # INSIGHTS INTO TEST DATATYPES
    print("test datatypes\n", test.dtypes, "\n")