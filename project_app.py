

import pandas as pd
import streamlit as st
import joblib

numerical_features=['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod']

lgm=joblib.load('lgm.pkl')
encoder=joblib.load('encoder.pkl')
scaler=joblib.load('scaler.pkl')
feature_names=joblib.load('feature_names.pkl')

st.set_page_config(page_title='Customer Churn Prediction App', layout='centered')

st.title('CUSTOMER CHURN PREDICTION APP')

st.write(
    'This application is a churn prediction system that identifies customers at risk of leaving.'
     'This will aid the management to take actions based on the prediction.'
)

st.subheader('⚠️Disclaimer')
st.info(
    'This application is for educational and demonstration purposes.'
    'The predictions do not guarantee business decision or professional assesment.'
    'Therefore should not be considered as business insight.'
)

st.divider()

st.subheader('👤Customer Information')

st.write('Enter the customers information')

gender=st.selectbox('Gender', ['Female', 'Male'])

senior_citizen=st.selectbox('Are a senior citizen?', ['No','Yes'])
senior_citizen_value=1.if senior_citizen=='Yes' else 0

partner=st.selectbox('Do you have a partner?', ['No','Yes'])

dependents=st.selectbox('Do you have dependents?', ['No','Yes'])

tenure=st.number_input('How many months have  you been with the  company?', min_value=0, max_value=100, value=12)

monthly_charges=st.number_input('Monthly charges', min_value=0.0, value=50.0)

total_charges=st.number_input('Total charges', min_value=0.0, value=600.0)

st.subheader('🌐Services')

phone_service=st.selectbox(
    'Do you have phone servive?',
    ['No', 'Yes']
)

multiple_lines=st.selectbox(
    'Do you have multiple phone lines?',
    ['No', 'Yes', 'No phone service']
)

internet_service=st.selectbox(
    'What type of internet service do you have?',
    ['DSL', 'Fiber optic', 'No internet service']
)

online_security=st.selectbox(
    'Do you have online security?',
    ['No', 'Yes', 'No internet service']
)

online_backup=st.selectbox(
    'Do you have online backup?',
    ['Yes', 'No', 'No internet service']
)

device_protection=st.selectbox(
    'Do you have device protection?',
    ['Yes', 'No', 'No internet service']
)

tech_support=st.selectbox(
    'Do you have technical support?',
    ['Yes', 'No', 'No internet service']
)

streaming_tv=st.selectbox(
    'Do you have a TV streaming service?',
    ['Yes', 'No', 'No internet service']
)

streaming_movies=st.selectbox(
    'Do you have a movie streaming service?',
    ['Yes', 'No', 'No internet service']
)

st.subheader('💳Contract and Billing')

contract=st.selectbox(
    'What type of contract do you have?',
    ['Month-to-month', 'One year', 'Two year']
)

paperless_billing=st.selectbox(
    'Do you use paperless billing?',
    ['No', 'Yes']
)

payment_method=st.selectbox( 
    'How do you make payments?',
    [
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)'
        ]
)

input_data=pd.DataFrame({
    'gender':[gender],
    'SeniorCitizen':[senior_citizen_value],
    'Partner':[partner],
    'Dependents':[dependents],
    'tenure':[tenure],
    'PhoneService':[phone_service],
    'MultipleLines':[multiple_lines],
    'InternetService':[internet_service],
    'OnlineSecurity':[online_security],
    'OnlineBackup':[online_backup],
    'DeviceProtection':[device_protection],
    'TechSupport':[tech_support],
    'StreamingTV':[streaming_tv],
    'StreamingMovies':[streaming_movies],
    'Contract':[contract],
    'PaperlessBilling':[paperless_billing],
    'PaymentMethod':[payment_method],
    'MonthlyCharges':[monthly_charges],
    'TotalCharges':[total_charges]
})

input_categorical=input_data[categorical_features]

input_numerical=input_data[numerical_features]

input_categorical_encoded=encoder.transform(input_categorical)
input_categorical_encoded=pd.DataFrame(
    input_categorical_encoded.toarray(),
    columns=encoder.get_feature_names_out(categorical_features)
)

input_numerical_scaled=scaler.transform(input_numerical)

input_numerical_scaled=pd.DataFrame(
    input_numerical_scaled,
    columns=numerical_features,
)

input_final=pd.concat(
    [input_numerical_scaled, input_categorical_encoded],
    axis=1
)

input_final=input_final[feature_names]


st.divider()

if st.button('Predict Churn', use_container_width=True):
    prediction=lgm.predict(input_final)[0]

    probability=lgm.predict_proba(input_final)[0][1]

    if prediction==1:
        st.error('🚩🚨The customer is likely to churn.')
        st.write(
            f'Estimated probability of churn:'
            f'**{probability*100:.2f}%**'
        )

    else:
        st.success('🏳️✅The customer is likely to stay.')
        st.write(
            f'Estimated probability of churn:'
            f'**{probability*100:.2f}%**'
        )
