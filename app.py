from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        job = float(request.form['job'])
        marital= float(request.form['marital'])
        education = float(request.form['education'])
        default= float(request.form['default'])
        housing= float(request.form['housing'])
        loan = float(request.form['loan'])
        contact = float(request.form['contact'])
        month= float(request.form['month'])
        day= float(request.form['day_of_week'])
        duration = float(request.form['duration'])
        empvrate=float(request.form['emp.var.rate'])
        cons=float(request.form['cons.price.idx'])
        consf=float(request.form['cons.conf.idx'])
        eur=float(request.form['euribor3m'])
        nremp=float(request.form['nr.employed'])
        campaign= float(request.form['campaign'])
        pdays=float(request.form['pdays'])
        previous= float(request.form['previous'])
        poutcome=float(request.form['poutcome'])
        if age<=32:
            nage=1
        elif age>32 and age<=47:
            nage=2
        elif age>47 and age<=70:
            nage=3
        elif age>70 and age<=98:
            nage=4

        if duration<=102:
            nduration=1
        elif duration>102 and duration<=180:
            nduration=2
        elif duration>180 and duration<=319:
            nduration=3
        elif duration>319 and duration<=644:
            nduration=4
        elif duration>644:
            nduration=5

        pred_args = [nage,job,marital,education,default,housing,loan,contact,day,month,nduration,campaign,previous,poutcome,pdays,nremp,eur,consf,empvrate,cons]
        
        ss = open("standard_scaler.pkl",'rb')
        ssc=joblib.load(ss)
   
        pred_args=ssc.transform([pred_args])

        mul_reg = open('model.pkl','rb')
        ml_model = joblib.load(mul_reg)
        model_predcition = ml_model.predict(pred_args)
        if model_predcition == 1:
            res = 'buy'
        else:
            res = 'not buy'
        #return res
    return render_template('predict.html', prediction =res)

if __name__ == '__main__':
    app.run(debug=True)