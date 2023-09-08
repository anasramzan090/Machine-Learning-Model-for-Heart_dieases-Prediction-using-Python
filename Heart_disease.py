#This program is written by M Anas Ramzan, as a project of internship supervised by Technohachs edu tceh
#importing important libraries to be used in program
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
#getting file containing data set
ds=pd.read_excel('D://ML Programs//Intern_Projects//heart_disease_detection//heart_disease_data.xlsx')
#visualizing the data
print(ds.shape)
print(ds.head())
print(ds.isnull().sum()) #checking if some missing vale, if there is then we will take mean of that coilomn and put it thare
print(ds.info())
print(ds['target'].value_counts())
#makinh x and y variables
X=ds.drop('target',axis=1)
Y=ds['target']
#saperating data for training and testing
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=3)
#training the model
model=LogisticRegression().fit(X_train,Y_train)
#Reading input from the user
input_str = input("Enter 13 numbers from data saperated by comma i.e:- 51,0,2,140,308,0,0,142,0,1.5,2,1,2: ")
#Spliting the input string into individual numbers
numbers_str = input_str.split(',')
#Converting the strings to floating-point values
numbers = [float(num_str) for num_str in numbers_str]
# Creating a list containing these numbers
input_list = numbers
#function for model prediction
def predict_with_model(in_data):
    id_as_np=np.asarray(in_data)
    inp_reshape=id_as_np.reshape(1,-1)
    prediction=model.predict(inp_reshape)
    return prediction
# Calling the model function with the input data
prediction_result = predict_with_model(input_list)
print("Input List:", input_list)
if prediction_result==0:
    print("Patient has NO Heart Disease")
elif prediction_result==1:
    print("Person is Heart Patient!!!")
print("Model Prediction Result:", prediction_result)
#Deploying machine learning trained model
filename='trained_model_for_heart_disease_prediction.sav'
pickle.dump(model,open(filename,'wb')) #dump to save model
loaded_model=pickle.load(open('trained_model_for_heart_disease_prediction.sav','rb'))
