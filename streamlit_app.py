import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open('model.pkl','rb'))
col_names_test = ['Age','Gender','TB','DB','Alkphos','Sgpt','Sgot','TP','ALB','A/G']

st.header('ML Prediction of Indian Liver Patients')

st.text('This ML model predicts whether a person is liver patient.')


age = st.number_input('Enter Age',value=65)
gender = st.selectbox('Enter Gender',('Male', 'Female'))
tb = st.number_input('Total Billirubin (TB)',value = 7.3)
db = st.number_input('Direct Billirubin',value = 4.1)
al = st.number_input('Alkphos',value = 490)
sgp = st.number_input('Sgpt',value = 60)
sgo = st.number_input('Sgot', value = 68)
tp = st.number_input('Total Protiens', value = 7)
alb = st.number_input('Albumin',value = 3.3)
ag = st.number_input('A/G:', value = 0.89)

#Button Creation
col1, col2 = st.columns(2)
with col1:
    if st.button('Check your result'):
        features=[age,gender, tb, db, al, sgp, sgo,tp, alb, ag]
        final_features = pd.DataFrame([features],columns=col_names_test)
        prediction = model.predict(final_features)
        if prediction == 1:
            st.write('You seems more likely to be a liver patient.')
        else:
            st.write('Congratulations.!! You seems unlikely to be a liver patient')

with col2:    
    if st.button('Clear the result'):
        st.write(' ')