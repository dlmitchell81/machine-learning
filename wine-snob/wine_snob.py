import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor

# STEP 1: Load the data
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')

# print first 5 rows of data
print(data.head())
# print no. of samples and no. of features
print(data.shape)
# summary statistics
print(data.describe())

# STEP 2: split data into training and test sets
y = data.quality
X = data.drop('quality', axis=1)

# 20% of the data as a test set for evaluating our model
# set an arbitrary "random state" (a.k.a. seed) so that we can reproduce our results
# it's good practice to stratify your sample by the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y,
                          test_size=0.2,
                          random_state=123,
                          stratify=y)

# STEP 3: Declare data preprocessing steps - standardisation
# Standardization is the process of subtracting the means 
# from each feature and then dividing by the feature standard deviations

# We use the Transformer API which allows you to "fit" a preprocessing 
# step using the training data the same way you'd fit a model...
# ...and then use the same transformation on future data sets!

# 1 Fit the transformer on the training set (saving the means and standard deviations)
scaler = preprocessing.StandardScaler().fit(X_train)

# 2 Apply the transformer to the training set (scaling the training data)
X_train_scaled = scaler.transform(X_train)
print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))

# 3 Apply the transformer to the test set (using the same means and standard deviations)
X_test_scaled = scaler.transform(X_test)
print(X_test_scaled.mean(axis=0))
print(X_test_scaled.std(axis=0))

pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100))
print(pipeline.get_params())

# STEP 4: Declare hyperparameters to tune
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

# STEP 5: Tune model using a cross-validation pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
# Fit and tune model
clf.fit(X_train, y_train)
print(clf.best_params_)

# STEP 6: Refit on the entire training set 
print(clf.refit)

# STEP 7: Evaluate model pipeline on test data
# Predict on a new set of data
y_pred = clf.predict(X_test)

# Now we can use the metrics we imported earlier to evaluate our model performance.
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

# STEP 8: Save model for future use
joblib.dump(clf, 'rf_regressor.pkl')
clf2 = joblib.load('rf_regressor.pkl')

# Predict data set using loaded model
clf2.predict(X_test)
