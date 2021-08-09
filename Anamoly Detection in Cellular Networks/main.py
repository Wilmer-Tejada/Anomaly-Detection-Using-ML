# importing packages
import pandas as pd # data processing
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization
from termcolor import colored as cl # text customization
import itertools # advanced tools

from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.model_selection import train_test_split # data split
from sklearn.tree import DecisionTreeClassifier # Decision tree algorithm
from sklearn.neighbors import KNeighborsClassifier # KNN algorithm
from sklearn.linear_model import LogisticRegression # Logistic regression algorithm
from sklearn.svm import SVC # SVM algorithm
from sklearn.ensemble import RandomForestClassifier # Random forest tree algorithm
from xgboost import XGBClassifier # XGBoost algorithm

from sklearn.metrics import confusion_matrix # evaluation metric
from sklearn.metrics import accuracy_score # evaluation metric
from sklearn.metrics import f1_score # evaluation metric




df = pd.read_csv("data/ML-MATT-CompetitionQT1920_train.csv",engine='python')
# df = pd.read_csv("data/ML-MATT-CompetitionQT1920_train.csv",engine='python', encoding_errors='ignore')

# Look at the data types
df.describe()
df.values
# Look at shape
df.dtypes

# Look at number of duplicate rows
duplicate_rows = df[df.duplicated()]
duplicate_rows.shape