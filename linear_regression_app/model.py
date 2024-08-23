import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data():
    california_housing = fetch_california_housing()
    X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    y = pd.Series(california_housing.target, name='MedHouseVal')
    return X, y

def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, rmse, r2



# Added these sections

# def predict_price(features, scaler, model):
#     features_scaled = scaler.transform([features])
#     prediction = model.predict(features_scaled)
#     return prediction[0]


def predict_price(model, scaler, input_data):
    # Transform the input data
    input_data_scaled = scaler.transform([input_data])
    # Predict price
    prediction = model.predict(input_data_scaled)
    return prediction[0]



# Prepare the model and scaler once at start
X, y = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
model = train_model(X_train, y_train)