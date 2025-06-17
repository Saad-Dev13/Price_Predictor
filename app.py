from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model and encoder
model = joblib.load('price_predictor.pkl')
mlb = joblib.load('mlb.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    modules = data.get('modules', [])
    days = data.get('days', 1)
    # One-hot encode modules
    modules_encoded = mlb.transform([modules])
    features = np.concatenate([modules_encoded[0], [days]])
    price = model.predict([features])[0]
    return jsonify({'predicted_price': float(price)})

if __name__ == '__main__':
    app.run(debug=True)