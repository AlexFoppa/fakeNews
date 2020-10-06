#Impory the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import pickle

#load data
originalDataset = pd.read_csv('news_articles.csv')
textData = originalDataset.iloc[:,10]
fakeOrReal = originalDataset.iloc[:,8]
dataset = pd.concat([textData, fakeOrReal], axis=1, sort=False)
dataset.drop(dataset.tail(50).index,inplace=True)

    
#creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(dataset.iloc[:,0]).toarray()
y = dataset.iloc[:,-1]

#naive bayes
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'gini', max_depth = 10)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Save file
filename = 'model04_randomForest.sav'
pickle.dump(classifier, open(filename, 'wb'))

loaded_model = pickle.load(open('model08_randomForest_mf1500_t1000.sav', 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)