import os

from flask import Flask, request, jsonify
import joblib
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('heart_model.h5')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json(force=True)
#     data_scaled = scaler.transform(np.array([list(data.values())]))
#     prediction = model.predict(data_scaled)
#     return jsonify({'prediction': int(prediction[0][0] > 0.5)})

@app.route("/", methods=["GET"])
def main():
    return "Welcome to Our Vardiocascular Predict API"

@app.route("/predict", methods=["POST"])
def predict():
    # Get incoming JSON data
    data = request.get_json(force=True)
    
    # Assign each feature to its own variable
    age = data.get('age')
    gender = data.get('gender')
    height = data.get('height')
    weight = data.get('weight')
    ap_hi = data.get('ap_hi')
    ap_lo = data.get('ap_lo')
    cholesterol = data.get('cholesterol')
    gluc = data.get('gluc')
    smoke = data.get('smoke')
    alco = data.get('alco')
    active = data.get('active')
    
    # Construct the input data as an array using the variables
    input_data = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
    # input_data = np.array([[55,1,164,62,140,80,1,1,0,0,0]])
    
    # Scale the input data
    data_scaled = scaler.transform(input_data)
    
    # Make a prediction
    prediction = model.predict(data_scaled)
    
    # Convert the prediction to a boolean (1 or 0) and return as a response
    prediction_bool = int(prediction[0][0] > 0.5)
    
    return jsonify({'prediction': prediction_bool})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
#    app.run(debug=True)
