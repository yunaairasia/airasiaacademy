import streamlit as st
import pandas as pd
import pickle

st.write("""
# Sale Prediction App

This app predicts the **Sale** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 0.700000, 	296.400000, 147.042500)
    Radio = st.sidebar.slider('Radio', 0.000000, 49.600000, 23.264000)
    Newspaper = st.sidebar.slider('Newspaper', 0.300000, 114.000000, 30.554000)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("yuna_model.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write(prediction)
