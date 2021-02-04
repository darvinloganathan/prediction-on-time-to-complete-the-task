import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import  LinearRegression
#import dataset
d=pd.read_excel('project basic.xlsx',sheet_name='Sheet1')
#getting numerical variable to preprocess
numeric_features = ['functionality in screen','no of project ','total experience','no of qa bugs','no of back log']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
#getting categorical variable to preprocess
categorical_features = ['requirement change ','work type','support']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
#data preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
#build linear regression  model 
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier',  LinearRegression())])

x=d[['functionality in screen','no of project ', 'total experience', 'requirement change ',
       'no of qa bugs', 'work type', 'no of back log', 'support']]
y=d['act_time(in hrs)']
model.fit(x,y)
pred=model.predict(x)
abs(pred-y).mean()

#saving the model
import pickle
with open ('linear_model1','wb') as file:
    pickle.dump(model,file)
