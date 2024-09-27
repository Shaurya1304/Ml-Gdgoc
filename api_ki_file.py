from flask import Flask,request, render_template
import numpy as np
import pickle
import sklearn
print(sklearn.__version__)
#loading models
dtr = pickle.load(open('ForDdgoc.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

#flask app
app = Flask(__name__)

@app.route('/')
def fend():
    return render_template('fend.html')
@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        NC = request.form['NC']
        pc = request.form['pc']
        kC = request.form['kc']
        avg_temp = request.form['avg_temp']
        avg_hum = request.form['avg_hum']
        pH = request.form['pH']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        Area = request.form['Area']
        price  = request.form['price']

        features = np.array([[NC,pc,kC,avg_temp,avg_hum,pH,average_rain_fall_mm_per_year,Area,price]],dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1,-1)

        return render_template('fend.html',prediction = prediction)

if __name__=="__main__":
    app.run(debug=True)