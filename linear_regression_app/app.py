from flask import Flask, render_template, request, jsonify
from model import load_data, preprocess_data, train_model, predict_price

app = Flask(__name__)

# Load and preprocess the data
X, y = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
model = train_model(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        features = [
            float(data['longitude']),
            float(data['latitude']),
            float(data['housing_median_age']),
            float(data['total_rooms']),
            float(data['total_bedrooms']),
            float(data['population']),
            float(data['households']),
            float(data['median_income'])
        ]
        prediction = predict_price(model, scaler, features)
        return jsonify({'price': prediction})
    except ValueError:
        return jsonify({'error': 'Invalid input data'}), 400

if __name__ == '__main__':
    app.run(debug=True)
