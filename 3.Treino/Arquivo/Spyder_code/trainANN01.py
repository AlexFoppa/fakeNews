#Impory the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import pickle

#load data
originalDataset = pd.read_csv('news_articles.csv')
textData = originalDataset.iloc[:,10]
fakeOrReal = originalDataset.iloc[:,8]
fakeOrReal = fakeOrReal.replace(['Real'],1)
fakeOrReal = fakeOrReal.replace(['Fake'],0)
dataset = pd.concat([textData, fakeOrReal], axis=1, sort=False)
dataset.drop(dataset.tail(50).index,inplace=True)
    
#creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(dataset.iloc[:,0]).toarray()
y = dataset.iloc[:,-1]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(X_train)
x_test = sc_x.transform(X_test)

#Ann building
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units = 64, kernel_initializer = 'random_normal' , activation = 'relu', input_dim = 1500))
classifier.add(Dense(units = 64, kernel_initializer = 'random_normal' , activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform' , activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 200)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.4)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



filename = 'model12_ann3.sav'
pickle.dump(classifier, open(filename, 'wb'))

print(cm)
