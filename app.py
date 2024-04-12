import os
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template("website.html")


@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict(final)
    output=prediction[0]

    if output==0:
        return render_template('predict.html',pred='You don\'t have any heart disease. Woohoo!!')
    else:
        return render_template('predict.html',pred='You may have heart disease. Please consult with the doctors.')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
