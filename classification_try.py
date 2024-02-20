import pandas as pd
import streamlit as st
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

with open('knn.pkl', 'rb') as file:
    rfc = pickle.load(file)

st.title("Classifying Bookings")


st.subheader("Choose an option")
option = st.radio(
    "Choose an option ",
    ("Choose file","Input value")
)


if option == "Input value":
    st.subheader("Provide input values")
    col1, col2, col3 = st.columns(3)
    with col1:
        input1 = st.text_input('Booking ID')
        input2 = st.number_input('No of Adults',min_value=0)
        input3 = st.number_input('No of Children',min_value=0)
        input4 = st.number_input('No of weekend nights',min_value=0)
        input5 = st.number_input('No of week nights',min_value=0)
        #input6 = st.number_input('Type of Meal Plan')
        input7 = st.number_input('Required car parking space', min_value=0)
        #input8 = st.number_input('Room type reserved')
        input9 = st.number_input('Lead time', min_value=0)
        input6 = st.selectbox('Type of Meal Plan', ('Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'))
        input8 = st.selectbox('Room type reserved', ('Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'))
        

    with col3:
        input10 = st.number_input('Arrival year', min_value = 2017, max_value = 2040)
        input11 = st.number_input('Arrival month', min_value=1, max_value=12)
        input12 = st.number_input('Arrival date', min_value=1, max_value=31)
        #input13 = st.number_input('Market Segment Type')
        input14 = st.number_input('Repeated Guest',min_value=0)
        input15 = st.number_input('No of previous cancellations',min_value=0)
        input16 = st.number_input('No of previous bookings not cancelled', min_value=0)
        input17 = st.number_input('Average price per room')
        input18 = st.number_input('No of special request',min_value=0)
        input13 = st.selectbox('Market Segment Type', ('Aviation', 'Complementary', 'Corporate', 'Offline', 'Online'))
    if input6=="Meal Plan 1":
        input6=0
    elif input6=="Meal Plan 2":
        input6=1
    elif input6=="Meal Plan 3":
        input6=2
    elif input6=="Not Selected":
        input6=3

    if input8=="Room_Type 1":
        input8=0
    elif input8=="Room_Type 2":
        input8=1
    elif input8=="Room_Type 3":
        input8=2
    elif input8=="Room_Type 4":
        input8=3
    elif input8=="Room_Type 5":
        input8=4
    elif input8=="Room_Type 6":
        input8=5
    elif input8=="Room_Type 7":
        input8=6
    
    if input13=="Aviation":
        input13=0
    elif input13=="Complementary":
        input13=1
    elif input13=="Corporate":
        input13=2
    elif input13=="Offline":
        input13=3
    elif input13=="Online":
        input13=4
    
    continue_button2= st.button("Proceed")
    if continue_button2:

        # Make predictions using the loaded model
        prediction = rfc.predict([[ input2, input3, input4, input5, input6,  input7,input8, input9,input11, input13,
                                input14, input15, input16, input17, input18]])

        # Display the prediction
        st.subheader("Prediction")
        if prediction == 0:
            st.write('Canceled')
        elif prediction == 1:
            st.write('Non_Canceled')

if __name__ == '__main__':
    main()

