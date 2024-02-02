# -*- coding: utf-8 -*-

import pickle
import time
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
st.title('TELECOMMUNICATION CHURN PREDICTION')
with st.sidebar:
    st.write('''Customer churn is a big problem for telecommunications companies. 
            Indeed, their annual churn rates are usually higher than 10%.This is a classification 
            project since the variable to be predicted is binary (churn or loyal customer). 
            The goal here is to model churn probability, conditioned on the customer features.''')

col1, col2, col3 = st.columns(3)
# Load the model
load = open('model.pkl', 'rb')
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
