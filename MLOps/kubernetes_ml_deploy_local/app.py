from flask import Flask, request, jsonify
import joblib
import numpy as np  

model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "ML model server running"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1) #reshape(1, -1) is used to convert the features into a 2D array with a single row
    prediction = model.predict(features)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
    