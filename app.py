# -*- coding: utf-8 -*-

import pickle
import time
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
st.title('_*:red[TELECOMMUNICATION CHURN PREDICTION]*_')
with st.container():
    with st.sidebar:
        st.header('_*:red[TELECOMMUNICATION CHURN PREDICTION]*_',  divider='rainbow')
        st.write('**MR. OMKAR SUNILDATT FAND**')
        st.page_link("https://www.linkedin.com/in/omkar-s-fand-043755149", label=":blue[LinkedIn]")
        st.page_link("http://www.gmail.com/", label=":blue[E-mail: omkarfand77@gmail.com]")
        st.subheader('Objective', divider='rainbow')
        st.write('''Customer churn is a big problem for telecommunications companies. 
            Indeed, their annual churn rates are usually higher than 10%.This is a classification 
            project since the variable to be predicted is binary (churn or loyal customer). 
            The goal here is to model churn probability, conditioned on the customer features.''')
        st.subheader('Dataset Features', divider='rainbow')
        st.write('''The dataset includes features such as state, area code, account length, voice plan, 
        voice messages, international plan, international minutes, international calls, international charge, 
        day minutes, day calls, day charge, evening minutes, evening calls, evening charge, night minutes, 
        night calls, night charge, customer service calls, and churn status.''')
        st.subheader('Steps Involved', divider='rainbow')
        st.write('''

    1. **Exploratory Data Analysis (EDA) and Preprocessing:** Understand data patterns and clean/preprocess the dataset.
    2. **Feature Engineering and Selection:** Create new features and select relevant ones. Scale the data and handle dependent/independent features.
    3. **Model Building:** Utilize the Random Forest Classifier algorithm to construct a predictive model.
    4. **Model Evaluation:** Achieve a model accuracy of 95%.
    5. **Deployment:** Deploy the model using the Streamlit framework for real-time application.''')
        st.subheader('Tools & Librares Used', divider='rainbow')
        st.write('Pandas, Numpy, Scikit-Learn (Classification Algorithms), Matplotlib, Seaborn, Streamlit')

col1, col2, col3 = st.columns(3)
# Load the model
with open('model.pkl', 'rb') as load:
    model = pickle.load(load)


# Imputer (you need to define this based on how you handled missing values during model training)
# Example: imputer = SomeImputer()
# Replace 'SomeImputer()' with the actual imputer you used during training.


def predict(ac, voice_plan, voice_msg, int_plan, int_min, int_call, int_chrg, day_min, day_call,
            day_chrg, eve_min, eve_call, eve_chrg, ni_min, ni_call, ni_chrg, customer_call):
    voice_plan = 1 if voice_plan == 'Yes' else 0
    int_plan = 1 if int_plan == 'Yes' else 0
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'account.length': [ac],
        'voice.plan': [voice_plan],
        'voice.messages': [voice_msg],
        'intl.plan': [int_plan],
        'intl.mins': [int_min],
        'intl.calls': [int_call],
        'intl.charge': [int_chrg],
        'day.mins': [day_min],
        'day.calls': [day_call],
        'day.charge': [day_chrg],
        'eve.mins': [eve_min],
        'eve.calls': [eve_call],
        'eve.charge': [eve_chrg],
        'night.mins': [ni_min],
        'night.calls': [ni_call],
        'night.charge': [ni_chrg],
        'customer.calls': [customer_call]})
    # Make prediction
    prediction = model.predict(input_data)
    result_message = 'Churn' if prediction[0] == 1 else 'Not Churn'
    return result_message


def main():
    with col1:
    # Input fields
        ac = st.number_input('account.length: ')
        voice_plan = st.selectbox('Have Voice Plan?', [' ', 'Yes', 'No'])
        voice_msg = st.number_input('voice.messages: ')
        int_plan = st.selectbox('Have International Plan?', (' ', 'Yes', 'No'))
        int_min = st.number_input('intl.mins: ')
        int_call = st.number_input('intl.calls: ')
    with col2:
        int_chrg = st.number_input('intl.charge: ')
        day_min = st.number_input('day.mins: ')
        day_call = st.number_input('day.calls: ')
        day_chrg = st.number_input('day.charge: ')
        eve_min = st.number_input('eve.mins: ')
        eve_call = st.number_input('eve.calls: ')
    with col3:
        eve_chrg = st.number_input('eve.charge: ')
        ni_min = st.number_input('night.mins: ')
        ni_call = st.number_input('night.calls: ')
        ni_chrg = st.number_input('night.charge: ')
        customer_call = st.number_input('customer.calls: ')

    # Button Logic
    if st.button('Predict'):
        # Call the prediction function
        result = predict(ac, voice_plan, voice_msg, int_plan, int_min, int_call, int_chrg, day_min, day_call,
                            day_chrg, eve_min, eve_call, eve_chrg, ni_min, ni_call, ni_chrg, customer_call)
        st.success(f'Churn Prediction: {result}')

if __name__ == '__main__':
    main()
