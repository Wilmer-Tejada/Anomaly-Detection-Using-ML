#!/usr/bin/env python
# coding: utf-8

# # Importing Necessary Packages
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
from sklearn.preprocessing import LabelEncoder # label encoding

from sklearn.metrics import confusion_matrix # evaluation metric
from sklearn.metrics import accuracy_score # evaluation metric
from sklearn.metrics import f1_score # evaluation metric
from sklearn.metrics import precision_score # evaluation metric
from sklearn.metrics import recall_score # evaluation metric
# # Importing Data
# This dataset was gethered from Kaggle:
# https://www.kaggle.com/c/anomaly-detection-in-cellular-networks/data
df = pd.read_csv("data/ML-MATT-CompetitionQT1920_train.csv",encoding='iso-8859-1')
df.shape
# Let's make sure we do not have duplicates to maintain accuracy.
df = df.drop_duplicates()
df.shape
df.head(5)
# Nothing looks out of the ordinary at first glance.
# # Cleaning Data
# First lets make sure that we imported all the columns in correctly.
df.dtypes
# It looks like all of the data types are correct except for the 'maxUE_UL+DL' column. It should be a float64 object so lets investigate that further.
# ### Fix df["maxUE_UL+DL"]
# In[6]:
df["maxUE_UL+DL"].unique()


# By looking at all the unique values in this column we see an unwanted value of #¡VALOR! which is what is coercing this numerical value into an object data type. Lets see what is happening by looking at some of the rows with this value. 

# In[7]:


issue_values = df[df["maxUE_UL+DL"] == '#¡VALOR!']
issue_values.shape


# In[8]:


# 84 rows of this incorrect value.
issue_values.head(20)
# Looks like '#¡VALOR!' should be nan. 


# Lets convert these incorrect values into nan.

# In[9]:


df["maxUE_UL+DL"] = pd.to_numeric(df["maxUE_UL+DL"], errors='coerce')
df["maxUE_UL+DL"].unique()


# After fixing our datatypes, let's make sure that all of our data types are correct

# In[10]:


df.dtypes


# Lets make all the columns numeric in order to feed the dataset through our algorithms. 

# 

# In[ ]:





# Lets look at some summary statistics to see if we have outliers

# In[11]:


df.describe()


# # Remove outliers

# In[12]:


std_scaler = StandardScaler()
# fit and transform the data
df.iloc[:,3:13] = std_scaler.fit_transform(df.iloc[:,3:13])


# ## Lets look at the variable we are trying to predict and see what kind of data it is.

# In[13]:


df["Unusual"].unique()


# It looks like our column has two values (Yes, No) coded as 1 and 0. This is a binary classification problem and so we can proceed with some classification algorithms in mind. 

# # Normalize dataset

# In[14]:


# df = (df - df.mean()) / (df.max() - df.min())


# # Balance Dataset

# In[15]:


# Look to see if this dataset is unbalanced. 
cases = len(df)
normal_count = len(df[df.Unusual== 0])
unusual_count = len(df[df.Unusual == 1])
unusual_percentage = round(unusual_count/cases*100, 2)


# In[16]:


print(cl('UNUSUAL COUNT', attrs = ['bold']))
print(cl('--------------------------------------------', attrs = ['bold']))
print(cl('Total number of cases are {}'.format(cases), attrs = ['bold']))
print(cl('Number of Normal cases are {}'.format(normal_count), attrs = ['bold']))
print(cl('Number of Unusual cases are {}'.format(unusual_count), attrs = ['bold']))
print(cl('Percentage of unusual cases is {}'.format(unusual_percentage), attrs = ['bold']))
print(cl('--------------------------------------------', attrs = ['bold']))


# ## Encode Labels

# In[17]:


# Encode labels to numeric
labelencoder = LabelEncoder()
df["Time"] = labelencoder.fit_transform(df["Time"])
df["CellName"] = labelencoder.fit_transform(df["CellName"])


# # Feature Selection and Data Split

# In[18]:


# Fill NaN values with zero. Might have to try different values here.
df.fillna(0, inplace=True)

X = df.drop('Unusual', axis = 1).values
y = df['Unusual'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# Here we encode our string values into numerical datatypes.
# We also replace nan with 0.
# Create test and train sets.

# # MODELING

# 1. Decision Tree

tree_model = DecisionTreeClassifier(max_depth = 4, criterion = 'entropy')
tree_model.fit(X_train, y_train)
tree_yhat = tree_model.predict(X_test)
# print("score on test: " + str(tree_model.score(X_test, y_test)))
# print("score on train: "+ str(tree_model.score(X_train, y_train)))

# 2. K-Nearest Neighbors

n = 5
knn = KNeighborsClassifier(n_neighbors = n)
knn.fit(X_train, y_train)
knn_yhat = knn.predict(X_test)
# print("score on test: " + str(knn.score(X_test, y_test)))
# print("score on train: "+ str(knn.score(X_train, y_train)))

# 3. Logistic Regression

lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(X_train, y_train)
lr_yhat = lr.predict(X_test)
# print("score on test: " + str(lr.score(X_test, y_test)))
# print("score on train: "+ str(lr.score(X_train, y_train)))

# 5. Random Forest Tree

rf = RandomForestClassifier(max_depth = 4)
rf.fit(X_train, y_train)
rf_yhat = rf.predict(X_test)

# 6. XGBoost
#Gradient boosting is an approach where new models are created that predict the residuals or errors of prior models and then 
# added together to make the final prediction. It is called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models.

xgb = XGBClassifier(max_depth = 4, use_label_encoder=False)
xgb.fit(X_train, y_train)
xgb_yhat = xgb.predict(X_test)

# # Results 

# ### Accuracy
# 1. Accuracy score

print(cl('ACCURACY SCORE', attrs = ['bold']))
print(cl('--------------------------------------------------', attrs = ['bold']))
print(cl('Decision Tree = {}'.format(accuracy_score(y_test, tree_yhat)), attrs = ['bold']))
print(cl('--------------------------------------------------', attrs = ['bold']))
print(cl('KNN = {}'.format(accuracy_score(y_test, knn_yhat)), attrs = ['bold']))
print(cl('--------------------------------------------------', attrs = ['bold']))
print(cl('Logistic Regression = {}'.format(accuracy_score(y_test, lr_yhat)), attrs = ['bold'], color = 'red'))
print(cl('--------------------------------------------------', attrs = ['bold']))
# print(cl('Accuracy score of the SVM model is {}'.format(accuracy_score(y_test, svm_yhat)), attrs = ['bold']))
# print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('Random Forest Tree = {}'.format(accuracy_score(y_test, rf_yhat)), attrs = ['bold']))
print(cl('--------------------------------------------------', attrs = ['bold']))
print(cl('XGBoost = {}'.format(accuracy_score(y_test, xgb_yhat)), attrs = ['bold'], color = 'green'))
print(cl('--------------------------------------------------', attrs = ['bold']))

# So a random forest is an ensemble method, which means it is collection of numerous decision trees that are taken in parralel and then the output of each tree is summarized to give one single output.

# XGboost is also an ensemble, but the reason that it is so powerful is because rather than just combining the isolated classifiers, it uses the mechanism of uplifting the weights of misclassified data points in the preceding classifiers.

# ### Recall
# 2. Recall score
#The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. 
# The recall is intuitively the ability of the classifier to find all the positive samples.

print(cl('RECALL SCORE', attrs = ['bold']))
print(cl('--------------------------------------------------', attrs = ['bold']))
print(cl('Decision Tree = {}'.format(recall_score(y_test, tree_yhat)), attrs = ['bold']))
print(cl('--------------------------------------------------', attrs = ['bold']))
print(cl('KNN = {}'.format(recall_score(y_test, knn_yhat)), attrs = ['bold']))
print(cl('--------------------------------------------------', attrs = ['bold']))
print(cl('Logistic Regression = {}'.format(recall_score(y_test, lr_yhat)), attrs = ['bold']))
print(cl('--------------------------------------------------', attrs = ['bold']))
# print(cl('Recall score of the SVM model is {}'.format(recall_score(y_test, svm_yhat)), attrs = ['bold'], color = 'red'))
# print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('Random Forest Tree = {}'.format(recall_score(y_test, rf_yhat)), attrs = ['bold']))
print(cl('--------------------------------------------------', attrs = ['bold']))
print(cl('XGBoost = {}'.format(recall_score(y_test, xgb_yhat)), attrs = ['bold'], color = 'green'))
print(cl('--------------------------------------------------', attrs = ['bold']))


# ### Precision
# 3. Precision score


print(cl('PRECISION SCORE', attrs = ['bold']))
print(cl('--------------------------------------------------', attrs = ['bold']))
print(cl('Decision Tree = {}'.format(precision_score(y_test, tree_yhat)), attrs = ['bold']))
print(cl('--------------------------------------------------', attrs = ['bold']))
print(cl('KNN = {}'.format(precision_score(y_test, knn_yhat)), attrs = ['bold']))
print(cl('--------------------------------------------------', attrs = ['bold']))
print(cl('Logistic Regression = {}'.format(precision_score(y_test, lr_yhat)), attrs = ['bold']))
print(cl('--------------------------------------------------', attrs = ['bold']))
# print(cl('Precision score of the SVM model is {}'.format(precision_score(y_test, svm_yhat)), attrs = ['bold'], color = 'red'))
# print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('Random Forest Tree = {}'.format(precision_score(y_test, rf_yhat)), attrs = ['bold']))
print(cl('--------------------------------------------------', attrs = ['bold']))
print(cl('XGBoost = {}'.format(precision_score(y_test, xgb_yhat)), attrs = ['bold'], color = 'green'))
print(cl('--------------------------------------------------', attrs = ['bold']))

# 4. F1 score

print(cl('F1 SCORE', attrs = ['bold']))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('F1 score of the Decision Tree model is {}'.format(f1_score(y_test, tree_yhat)), attrs = ['bold']))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('F1 score of the KNN model is {}'.format(f1_score(y_test, knn_yhat)), attrs = ['bold']))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('F1 score of the Logistic Regression model is {}'.format(f1_score(y_test, lr_yhat)), attrs = ['bold']))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
# print(cl('F1 score of the SVM model is {}'.format(f1_score(y_test, svm_yhat)), attrs = ['bold'], color = 'red'))
# print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('F1 score of the Random Forest Tree model is {}'.format(f1_score(y_test, rf_yhat)), attrs = ['bold']))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('F1 score of the XGBoost model is {}'.format(f1_score(y_test, xgb_yhat)), attrs = ['bold'], color = 'green'))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))

# 3. Confusion Matrix

# defining the plot function

def plot_confusion_matrix(cm, classes, title, normalize = False, cmap = plt.cm.Blues):
    title = 'Confusion Matrix of {}'.format(title)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix for the models

tree_matrix = confusion_matrix(y_test, tree_yhat, labels = [0, 1]) # Decision Tree
knn_matrix = confusion_matrix(y_test, knn_yhat, labels = [0, 1]) # K-Nearest Neighbors
lr_matrix = confusion_matrix(y_test, lr_yhat, labels = [0, 1]) # Logistic Regression
# svm_matrix = confusion_matrix(y_test, svm_yhat, labels = [0, 1]) # Support Vector Machine
rf_matrix = confusion_matrix(y_test, rf_yhat, labels = [0, 1]) # Random Forest Tree
xgb_matrix = confusion_matrix(y_test, xgb_yhat, labels = [0, 1]) # XGBoost

# Plot the confusion matrix

plt.rcParams['figure.figsize'] = (6, 6)

# 1. Decision tree
tree_cm_plot = plot_confusion_matrix(tree_matrix, 
                                classes = ['Usual(0)','Unusual(1)'], 
                                normalize = False, title = 'Decision Tree')
plt.savefig('tree_cm_plot.png')
plt.show()

# 2. K-Nearest Neighbors
knn_cm_plot = plot_confusion_matrix(knn_matrix, 
                                classes = ['Usual(0)','Unusual(1)'], 
                                normalize = False, title = 'KNN')
plt.savefig('knn_cm_plot.png')
plt.show()

# 3. Logistic regression
lr_cm_plot = plot_confusion_matrix(lr_matrix, 
                                classes = ['Usual(0)','Unusual(1)'], 
                                normalize = False, title = 'Logistic Regression')
plt.savefig('lr_cm_plot.png')
plt.show()

# 5. Random forest tree
rf_cm_plot = plot_confusion_matrix(rf_matrix, 
                                classes = ['Usual(0)','Unusual(1)'], 
                                normalize = False, title = 'Random Forest Tree')
plt.savefig('rf_cm_plot.png')
plt.show()

# # 6. XGBoost
xgb_cm_plot = plot_confusion_matrix(xgb_matrix, 
                                classes = ['Usual(0)','Unusual(1)'], 
                                normalize = False, title = 'XGBoost')
plt.savefig('xgb_cm_plot.png')
plt.show()
# The XGBoost algorithm, short for Extreme Gradient Boosting, is simply an improvised version of the gradient boosting algorithm, and the working procedure of both is almost the same. One crucial point in XGBoost is that it implements parallel processing at the node level, making it more powerful and fast than the gradient boosting algorithm. XGBoost reduces overfitting and improves overall performance by including various regularization techniques by setting the hyperparameters of the XGBoost algorithm.