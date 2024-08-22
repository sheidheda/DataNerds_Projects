import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load the California Housing dataset
california_housing = fetch_california_housing()

# Select only the relevant features
X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)[['MedInc', 'HouseAge', 'AveRooms']]
y = pd.Series(california_housing.target, name='MedHouseVal')

housing = X

# Set up Streamlit page
st.set_page_config(page_title="DN Housing Predictor", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")
st.title("Housing Price Prediction 🏠")
st.write('Use the sliders on the left to make the housing prediction.')
st.write('---')

# Sidebar for user inputs
with st.sidebar:
    st.title("Select Your Preference")
    # Collect user input
    user_input = {
        'MedInc': st.slider('Median Income', housing['MedInc'].min(), housing['MedInc'].max(), housing['MedInc'].median()),
        'HouseAge': st.slider('Housing Median Age', housing['HouseAge'].min(), housing['HouseAge'].max(), housing['HouseAge'].median()),
        'AveRooms': st.slider('Average Rooms', housing['AveRooms'].min(), housing['AveRooms'].max(), housing['AveRooms'].median()),
    }

# Convert user input to DataFrame
user_input_df = pd.DataFrame([user_input])

# Feature scaling
scaler = StandardScaler()

# Fit the scaler to the training data and transform both X and the user input
X_scaled = scaler.fit_transform(X)
user_input_scaled = scaler.transform(user_input_df)

# Train the model on the scaled data
model = LinearRegression()
model.fit(X_scaled, y)


# Input parameters and prediction
st.markdown('### Prediction')

# Predict the price using the scaled user input
prediction = model.predict(user_input_scaled)[0]

# Ensure the prediction is non-negative
prediction = max(prediction, 0)

st.metric(label="Predicted Median House Value", value=f"${prediction * 1000:,.0f}")

with st.expander("Feature Distributions"):
    st.subheader("Feature Distributions")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(housing['MedInc'], bins=30, kde=True, ax=axes[0])
    axes[0].set_title('Median Income Distribution')
    sns.histplot(housing['HouseAge'], bins=30, kde=True, ax=axes[1])
    axes[1].set_title('House Age Distribution')
    sns.histplot(housing['AveRooms'], bins=30, kde=True, ax=axes[2])
    axes[2].set_title('Average Rooms Distribution')
    st.pyplot(fig)

# Generate the pairplot
st.subheader("Pairplot of Scaled Features")
fig = sns.pairplot(X)
st.pyplot(fig)

