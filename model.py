import pandas as pd
df=pd.read_excel('project basic.xlsx',sheet_name='Sheet1')

df.head()

from sklearn.linear_model import LinearRegression

lm=LinearRegression(fit_intercept=False)

df.columns

x=df[['functionality in screen','no of project ', 'total experience', 'requirement change ',
       'no of qa bugs', 'work type', 'no of back log', 'support']]
y=df['act_time(in hrs)']

model1=lm.fit(x,y)
pred=model1.predict(x)
mae=abs(y-pred).mean()

import pickle
with open ('linear_model','wb') as file:
    pickle.dump(model1,file)
    