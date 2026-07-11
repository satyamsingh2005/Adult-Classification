import joblib
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
model=joblib.load(open('best_model.pkl','rb'))
@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():

    data=request.json['data']
    print(data)
    df = pd.DataFrame([data])
    output=model.predict(df)[0]
    return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():

    data={
        "age" : int(request.form["age"]),
        "education_num": int(request.form["education_num"]),
        "capital_gain": int(request.form["capital_gain"]),
        "capital_loss": int(request.form["capital_loss"]),
        "hours_per_week": int(request.form["hours_per_week"]),
        "workclass" : request.form["workclass"].strip(),
        "marital_status" : request.form["marital_status"].strip(),
        "occupation": request.form["occupation"].strip(),
        "relationship": request.form["relationship"].strip(),
        "race": request.form["race"].strip(),
        "sex": request.form["sex"].strip()
    }
    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    if prediction == 1:
        result = "Income is likely ABOVE $50K"
        result_color = "#22c55e"   # Green
    else:
        result = "Income is likely BELOW or EQUAL TO $50K"
        result_color = "#ef4444"   # Red

    return render_template("home.html",prediction_text=result,confidence=f"{probability:.2%}",result_color=result_color)

if __name__=="__main__":
    app.run(debug=True)





