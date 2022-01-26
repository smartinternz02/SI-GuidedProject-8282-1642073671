import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv("diabetes.csv")
x = np.array(data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']])
y = np.array(data[['Outcome']])

rf = RandomForestClassifier(n_estimators=1000,random_state=50)
rf.fit(x,y.ravel())

model = pickle.dump(rf,open("diabetes.pkl",'wb'))