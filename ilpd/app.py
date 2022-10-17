from flask import Flask, render_template, redirect, request, url_for
from joblib import load
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
col_names_test = ['Age','Gender','TB','DB','Alkphos','Sgpt','Sgot','TP','ALB','A/G']

@app.route('/')
def home():
    return render_template("index_inwork.html")


@app.route('/result', methods=['GET'])
def result():
    features=[]
    for val in request.args.values():
        features.append(val)
    final_features = pd.DataFrame([features],columns=col_names_test)
    prediction = model.predict(final_features)
    if prediction == 1:
        prediction_text = "You seems more likely to be a liver patient. Please consult a doctor.!!"
    else:
        prediction_text = "Congratulations.!!!You seems unlikely to be a liver patient"

    return render_template("index_inwork.html", results=prediction_text)


app.run(debug=True)
