# Import required libraries
import pickle
import pandas as pd
# Import Dataset
cancer_df=pd.read_csv('D:/ICT TRAINING/MINI PROJECT/data.csv')
cancer_df.drop(['Unnamed: 32','id','perimeter_mean','area_mean','concavity_mean','concave points_mean','texture_se','perimeter_se','area_se','concavity_se','concave points_se','symmetry_se','perimeter_worst','area_worst','concavity_worst','concave points_worst',],axis=1,inplace=True)
cancer_df['diagnosis']=cancer_df['diagnosis'].map({'M':1,'B':0})
x=cancer_df.drop(['diagnosis'],axis=1)
y=cancer_df['diagnosis']
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
## Split the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=60)
## MODELLING
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(class_weight={0:0.4,1:0.6})
from sklearn.ensemble import BaggingClassifier
bc=BaggingClassifier(n_estimators=350,base_estimator=dt,random_state=60)
bc.fit(x_train,y_train)
#Saving the model to disk
pickle.dump(bc,open('model.pkl','wb')) 