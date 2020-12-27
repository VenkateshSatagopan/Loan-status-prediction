from flask import Flask, render_template,request,jsonify
import pandas as pd
import pickle
import numpy as np
import joblib
import pandas as pd
import joblib
import catboost
from sklearn.ensemble import RandomForestClassifier, VotingClassifier,GradientBoostingClassifier,ExtraTreesClassifier
import xgboost


model=joblib.load('best-model-loan-prediction.pkl')

app=Flask(__name__)


@app.route('/',methods=['GET','POST'])
def get_details():
    d = None
    if request.method=='POST':
        d = request.form.to_dict()
        df = pd.DataFrame([d.values()], columns=d.keys())
        #df['Unnamed: 0']=-1
        #print(df.info())
        df['Gender']=df['Gender'].astype('str')
        df['Married']=df['Married'].astype('str')
        df['Dependents']=df['Dependents'].astype('str')
        df['Education']=df['Education'].astype('str')
        df['Self_Employed']=df['Self_Employed'].astype('str')
        df['ApplicantIncome']=df['ApplicantIncome'].astype('float64')
        df['CoapplicantIncome']=df['CoapplicantIncome'].astype('float64')
        df['LoanAmount']=df['LoanAmount'].astype('float64')
        df['Loan_Amount_Term']=df['Loan_Amount_Term'].astype('float64')
        df['Credit_History']=df['Credit_History'].astype('float64')
        df['Property_Area']=df['Property_Area'].astype('str')
        #print(df)
        prediction_val=model.predict(df)
        #print(prediction_val)
        prediction_val=int(prediction_val)
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
    prediction_val = int(model.predict(data_unseen))
    if prediction_val==0:
        prediction='rejected'
    else:
        prediction='approved'
    
    return jsonify('The model predicts that the loan will be {}'.format(prediction))
    
    
if __name__=='__main__':
    app.run(debug=True)
