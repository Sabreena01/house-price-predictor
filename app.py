import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression # type: ignore

# Load the dataset
df = pd.read_csv("house_prices_sample.csv")

# Define features and target
X = df[['square_feet', 'bedrooms', 'bathrooms']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("🏠 House Price Predictor")
st.write("Enter the details below to estimate house price:")

# Inputs
sqft = st.number_input("Square Feet", min_value=100, max_value=10000, step=100)
bed = st.number_input("Bedrooms", min_value=1, max_value=10, step=1)
bath = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)

if st.button("Predict Price"):
    input_data = np.array([[sqft, bed, bath]])
    prediction = model.predict(input_data)
    st.success(f"Estimated Price: ₹ {int(prediction[0]):,}")
