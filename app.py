import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
regmodel = pickle.load(open('regressionmodel.pkl', 'rb'))

@app.route('/')
def home():

    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    new_data = np.array(list(data.values())).reshape(1,-1)
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0].astype(float))

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1,-1)
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template('home.html', prediction_text="Class is {}".format(output))
    




if __name__ == "__main__":
    app.run(debug=True)