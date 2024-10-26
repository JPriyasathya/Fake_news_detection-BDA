from flask import Flask, request, jsonify
import joblib

# Load the model and vectorizer
model = joblib.load('random_forest_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    X_vectorized = vectorizer.transform([text])
    prediction = model.predict(X_vectorized)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
