# Import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#for data splitting, sacling, and encoding
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# For dimensionality reduction 
from sklearn.decomposition import PCA

# For baseline models 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR 

# Model evaluation metrics 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset 
df = pd.read_csv('education_career_success.csv')

# Inspect the first few rows 
print(df.head())

# Get a concise summary of the DataFrame 
print(df.info())

# Genereate descriptive statistics 
print(df.describe())

