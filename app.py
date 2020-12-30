from flask import Flask, render_template,request,jsonify,redirect
import pandas as pd
import pickle
import numpy as np
import joblib
import pandas as pd
import joblib
import catboost
from sklearn.ensemble import RandomForestClassifier, VotingClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from werkzeug.utils import secure_filename
import xgboost
import os
import time

model=joblib.load('best-model-loan-prediction.pkl')

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home():
    if request.method=='POST':
        try: 
            if request.form['Single Data Prediction']=='single data':
                return redirect('single_data_prediction')
        except:
            pass
        try: 
            if request.form['Batch Data Prediction']=='batch data':
                return redirect('batch_data_prediction')
        except:
            pass

    return render_template('home.html')

@app.route('/batch_data_prediction',methods=['GET','POST'])
def bacth_prediction():
    if request.method =='POST':
        d=request.form.to_dict()
        
        csv_file=list(d.values())[0]
        print(csv_file)
        try:
            if csv_file:
              df=pd.read_csv(csv_file)
              
              df['prediction']=model.predict(df.drop(['Loan_ID'],axis=1))
              print(df.head())
              basepath = os.path.dirname(__file__)
              print(basepath)
              timestr = time.strftime("%Y%m%d-%H%M%S")
              final_csv_file='prediction_'+str(timestr)+'.csv'
              print(final_csv_file)
              file_path = os.path.join(basepath, 'static',  final_csv_file)
              print(file_path)
              df.to_csv(file_path,index=False)
              return render_template("index_1.html",prediction='The model prediction is done and the predicted results is saved in the path {} under the name {}'.format( os.path.join(basepath, 'static'),secure_filename(final_csv_file)))
        except Exception as e:
            return render_template("index_1.html",prediction='Please enter the csv file correctly')

    return render_template("index_1.html",prediction='No data has been provided')        

@app.route('/single_data_prediction',methods=['GET','POST'])
def get_details():
    d = None
    if request.method=='POST':
        d = request.form.to_dict()
        df = pd.DataFrame([d.values()], columns=d.keys())
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