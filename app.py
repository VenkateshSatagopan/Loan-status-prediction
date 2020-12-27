from flask import Flask, render_template,request,jsonify
import pandas as pd
import pickle
import numpy as np
from pycaret.classification import *

model=load_model('best-model-loan-prediction')

app=Flask(__name__)


@app.route('/',methods=['GET','POST'])
def get_details():
    d = None
    if request.method=='POST':
        d = request.form.to_dict()
        df = pd.DataFrame([d.values()], columns=d.keys())
        prediction_val=predict_model(model, data=df, round = 0)
        prediction_val=int(prediction_val.Label[0])
        if prediction_val==0:
            prediction='rejected'
        else:
            prediction='approved'
            
        return render_template("index.html",prediction='The model predicts that the loan will be {}'.format(prediction))

    return render_template("index.html",prediction="No data has been provided yet")

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls throught request
    '''
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)
    
    
if __name__=='__main__':
    app.run(debug=True,port=300)