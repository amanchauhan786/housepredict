import streamlit as st
import pickle
import numpy as np
import json

# load model
with open('bangalore_house_price_predictor_model.pickle', 'rb') as model_file:
    model = pickle.load(model_file)

# Load column names from JSON file
with open('columns.json', 'r') as json_file:
    columns_data = json.load(json_file)
    columns = columns_data.get('data_columns', [])


# print(columns)

def predict_price(location, sqft, bath, bhk):
    loc_index = columns.index(location) if location in columns else -1

    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]


st.title("House Price Predictor")

# User input for location
location = st.selectbox("Select Location", columns[3:])

# User input for sqft area
sqft = st.number_input("Sqft Area", min_value=100, max_value=5000, step=100, value=1500)

# User input for number of bathrooms
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2)

# User input for number of bedrooms (bhk)
bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=5, value=2)

# Button to trigger prediction
if st.button("Predict"):
    # Make prediction using the input values
    prediction = predict_price(location, sqft, bath, bhk)

    # Center-aligned and bigger font
    st.markdown(f"<p style='text-align:center; font-size:24px;'>The estimated price is: {prediction:,.2f} lacs</p>",
                unsafe_allow_html=True)

def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''

    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url("https://pure.wallpapergk.com/download/night_city_lights_buildings_architecture_night-1600x900.jpg");
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )
set_bg_hack_url()