# Anomaly Detection In Cellular Networks Using ML

The full code, along with more explanation and images can be found in the [python notebook.](https://github.com/Wilmer-Tejada/Anomaly_Detection_Using_ML/blob/main/Anamoly%20Detection%20in%20Cellular%20Networks/Anomaly_detection.ipynb)
.  
## Context:
Traditionally, the design of a cellular network focuses on the optimization of energy and resources that guarantees a smooth operation even during peak hours (i.e. periods with higher traffic load). However, this implies that cells are most of the time overprovisioned of radio resources. Next generation cellular networks ask for a dynamic management and configuration in order to adapt to the varying user demands in the most efficient way with regards to energy savings and utilization of frequency resources. If the network operator were capable of anticipating to those variations in the users’ traffic demands, a more efficient management of the scarce (and expensive) network resources would be possible. Current research in mobile networks looks upon Machine Learning (ML) techniques to help manage those resources. In this case, you will explore the possibilities of ML to detect abnormal behaviors in the utilization of the network that would motivate a change in the configuration of the base station.

## Problem
We need to be able to anticipate variations in the user’s traffic demands. This would allow for the efficient management of the scarce network resources. In this case we will attempt to predict anomalies in cellular networks using ML. 

## Steps Involved
1. Importing the required packages and data
2. Processing the data to our needs
3. Feature selection and data split
4. Building our classification models
5. Evaluating our models

## 1. Importing the required packages and data
Data set from Kaggle: https://www.kaggle.com/c/anomaly-detection-in-cellular-networks/data

Packages required:

- Pandas, NumPy, Scikit Learn, XGBoost, Seaborn

## 2. Processing the data to our needs
1. Remove any duplicate data (106 out of 36,904)
2. Fix incorrect labels and coerce data types.
3. Encode all strings into numeric values in order to feed them to our algorithms.  
4. Impute all missing data with mean value. 
5. Remove outliers from dataset.
6. Normalize all our data.
7. See if we need to rebalance our dataset.
8. Compare our features to our target variable.

### Correlation Matrix
![image](https://user-images.githubusercontent.com/18300911/128751323-b44bfbb0-ead0-49f7-8764-4de8d99605e3.png)
Low correlation between features and target variable implies a non-linear relationship. Tree-based methods based on entropy will probably outperform linear methods. 

## 3. Feature selection and data split
Because our dataset is small, we can keep all of our columns without any computational power worries. Otherwise, we could use dimensionality reduction techniques such as Principal Component Analysis or Recursive Feature Elimination. 
We split our dataset into 80% Train/ 20% Test.

## 4. Building our classification models
Because this is a binary classification problem, I chose to use the following algorithms: 
### - Decision Tree
### - K-nn
### - Log Regression
### - Random Forest
### - XGBoost

## 5. Evaluating our models
We can evaluate our model using a confusion matrix and the following four metrics. A confusion matrix is provided below for reference. 
- Accuracy = (tp + tn) / (tp + tn + fp + fn)
- Precision = tp / (tp + fp)
- Recall = tp / (tp + fn)
- F1 Score = 2((precision * recall) / (precision + recall))

## Confusion Matrix
![image](https://user-images.githubusercontent.com/18300911/128752226-98de7182-e265-4754-9946-a9233259076e.png)

## Accuracy
One of the most important metrics for evaluating our model is looking at the accuracy of predictions. This boils down to correct predictions/all predictions.
![image](https://user-images.githubusercontent.com/18300911/128752197-25d76510-3348-4bd3-b3f5-0fadd71180ca.png)

## Recall 
The recall is intuitively the ability of the classifier to find all the positive samples.
For example for a medical diagnosis, it is better to maximize recall here because we rather have somebody test positive for a diagnosis and them not actually have it, rather  than somebody test negative when they actually DO have a medical condition. when we increase the recall, we decrease the precision.
![image](https://user-images.githubusercontent.com/18300911/128752848-49e3bbc7-cbb9-42fe-97f5-482bd2b8ae30.png)


## Precision
Precision is ability of a classification model to identify only the relevant data points. Continuing with the medical diagnosis example, if this model labels everybody as having a certain diagnosis then it isn’t too precise of a model. 
![image](https://user-images.githubusercontent.com/18300911/128752874-7ca2c342-63db-4617-9c1d-dc38baaa0be9.png)


## F1 - Score
The F1 score can be interpreted as a weighted average of the precision and recall.
![image](https://user-images.githubusercontent.com/18300911/128752908-3371b3e4-d13e-4ddb-a3c7-42b3272ac3b5.png)


## Findings
The XGBoost model performs significantly better than most of the other models we tried. 
The final results we get are:

- Accuracy = 0.981
- Precision = 0.997
- Recall = 0.933
- F1 Score = 0.964

![image](https://user-images.githubusercontent.com/18300911/128753573-bd1d9794-ed32-46f5-ba75-389c5c04a263.png)


The simpler learners are not accurate enough on their own. Using an ensemble method like XGBoost is powerful is because rather than just combining the isolated classifiers, it uses the mechanism of uplifting the weights of misclassified data points in the preceding classifiers. With Boosting, new models are added to correct the errors made by existing models. Models are added sequentially until no further improvements can be made. 

## What’s next?
Because this model was built with python, it can be deployed anywhere python code can be deployed. This also includes AWS, GCP, Azure, etc. Using the pickle library simplifies this quite a bit as it allows you to store model weights and parameters.


 

