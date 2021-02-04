import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('linear_model1', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = pd.DataFrame([int_features],columns=['functionality in screen', 'no of project ', 'total experience',
       'requirement change ', 'no of qa bugs', 'work type', 'no of back log',
       'support'])
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('home.html', prediction_text='No of hours taken to complete the task is  {}'.format(output))


if __name__ == "__main__":
    app.run()
    

