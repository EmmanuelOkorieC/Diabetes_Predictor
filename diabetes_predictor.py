import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle as pk

diabetes_data = pd.read_csv('diabetes.csv')

#studying data
#print(diabetes_data.head())
#print(diabetes_data.shape)
#print(diabetes_data.describe())

#print(diabetes_data['Outcome'].value_counts())
#print(diabetes_data.groupby('Outcome').mean())

X = diabetes_data.drop(columns='Outcome', axis=1)
Y = diabetes_data['Outcome']


#Preprocessing
scaler = StandardScaler()
standardized_data = scaler.fit_transform(X)
X = standardized_data

#Train Test Split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


#Training the model
model= svm.SVC(kernel='linear')
model.fit(X_train,Y_train)

#training data accuracy
#training_prediction = model.predict(X_train)
#training_data_accuracy = accuracy_score(training_prediction, Y_train)
#print(training_data_accuracy)

#test data accuracy
#test_prediction = model.predict(X_test)
#test_data_accuracy = accuracy_score(test_prediction, Y_test)
#print(test_data_accuracy)

#Save model and scaler
with open('diabetes_model.pkl', 'wb') as f:
    pk.dump(model, f)

with open('scalar.pkl', 'wb') as f:
    pk.dump(scaler, f)

print('Model saved successfullay')