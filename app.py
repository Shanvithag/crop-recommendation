from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app= Flask(__name__)

model= pickle.load(open(r'C:\Users\deepu\Downloads\crop recommendation\model.pkl',"rb"))

@app.route('/services')
def services():
    return render_template('Services1.html')

@app.route('/predictpage')
def predictPage():
    return render_template('getrecommendation1.html')

@app.route('/homepage')
def homePage():
    return render_template('index1.html')

@app.route('/')
def home():
    return render_template('index1.html')

@app.route("/predict", methods=['POST'])
def predict():
    # Extract form data
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosphorus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    # Make prediction
    features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    prediction = model.predict(features)

    # Mapping prediction to crop names
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = f"{crop}."
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    # Render the prediction result back to the HTML template
    return render_template('getrecommendation1.html',result=result)


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

