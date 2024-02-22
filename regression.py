import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
import pickle

st.markdown("<h2 style='text-align: center; color: black;'> Spend Prediction </h2>", unsafe_allow_html=True)



with open("linear.pkl", 'rb') as file:
    model = pickle.load(file)


st.subheader("Choose an option")
option = st.radio(
    "Choose an option ",
    ("Upload file", "Input value")
)

if option == "Upload file":
    
    st.subheader("Upload file")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    #uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        df_copy=df.copy()
        # Display original data
        st.subheader("Original Data")
        st.write(df.head(5))
        df = df[[ 'tier', 'marital_status', 'leisure_stays',
       'business_stays', 'total_stays', 'length_of_stay', 'total_luxury_stays',
       'total_full_service_stays', 'email_open_rate', 'email_click_rate',
       'guest_count', 'customer_income', 'customer_income2',
       'international_travel_interest', 'domestic_travel_interest']]    

        #getting numerical columns
        def get_numerical_column(dataframe):
            return dataframe.columns[dataframe.dtypes != "object"]

        numerical_columns = get_numerical_column(df)
    
        def get_categorical_column(dataframe):
            return dataframe.columns[dataframe.dtypes == "object"]

        categorical_columns = get_categorical_column(df)

    
        def fill_missing_values_mean(columns_to_fill, dataframe):
            for col_name in columns_to_fill:
                mean = dataframe[col_name].mean()
                dataframe[col_name].fillna(mean,inplace=True)

        def missing_check(df):
            has_missing_values = df.isnull().values.any()
            #print("Status Missing Values:",has_missing_values)

            if has_missing_values:
                #print("\nMissing Values: ")
                #print(df.isnull().sum().sort_values(ascending = False))
                def fill_missing_values(df):
                #df[col].fillna(val, inplace=True)
                    df['marital_status'].fillna("U", inplace=True)
                    df['tier'].fillna("b", inplace=True)
                    df['international_travel_interest'].fillna(1, inplace=True)
                fill_missing_values(df)
                #print("After handling missing values:")
                #print(df.isnull().sum().sort_values(ascending = False))
            else:
                #print("\nNo Missing Values Found")
                pass
        missing_check(df_copy)
        missing_check(df)
    
        def duplicate_check(df):
            has_duplicates = df.duplicated().sum()
            #print("Total duplicates present : ", has_duplicates)
            if has_duplicates!=0:
                df = df.drop_duplicates()
            else:
                pass   
        duplicate_check(df)
        columns_to_encode = categorical_columns
        
        continue_button =st.button("Proceed")
        if continue_button:
            def encode_column(df, column):
                for i in column:
                    label = LabelEncoder()
                    df[i] = label.fit_transform(df[i])
            
            encode_column(df,columns_to_encode)
            
            
                #########################
            st.subheader("Predicted Data")
            #st.write(df.head(5))
            y_pred = model.predict(df)
            # Using DataFrame.insert() to add a column
            #df.insert("Predicted value", y_pred, True)

            df2 = df_copy.assign(Predicted=y_pred)
            #df2['Predicted'] = df2['Predict'].apply(lambda x: 'Canceled' if x == 0 else 'Not_Canceled')
            #df2 = df2.drop([''], axis = 1)

            st.write(df2.head(5))

if option == "Input value":
    st.subheader("Provide input values")
    col1, col2, col3 = st.columns(3)
    with col1:
        input1 = st.text_input('Customer ID')
        #input2 = st.number_input('tier')
        #input3 = st.number_input('Marital Status')
        input4 = st.number_input('Leisure stays',min_value=0)
        input5 = st.number_input('Business stays',min_value=0)
        input6 = st.number_input('Total stays',min_value=0)
        input7 = st.number_input('Length of stay',min_value=0)
        input8 = st.number_input('Total luxury stays',min_value=0)
        input9 = st.number_input('Total full service stays', min_value=0)
        input2 = st.selectbox('Tier', ('b', 'd', 'g', 's'))
            

    with col3:
        input10 = st.number_input('Email open rate',min_value=0.0, max_value=1.0)
        input11 = st.number_input('Email click rate',min_value=0.0,max_value=1.0)
        input12 = st.number_input('Guest count',min_value=0)
        input13 = st.number_input('Customer income',min_value=0)
        input14 = st.number_input('Customer income2', min_value=0)
        input15 = st.number_input('International travel interest',min_value=0)
        input16 = st.number_input('Domestic travel interest',min_value=0)
        input3 = st.selectbox('Marital Status', ('D', 'M', 'S', 'W'))
    
        #input13 = st.selectbox('Market Segment Type', ('Aviation', 'Complementary', 'Corporate', 'Offline', 'Online'))
    if input2=="b":
        input2=0
    elif input2=="d":
        input6=1
    elif input2=="g":
        input6=2
    elif input2=="s":
        input6=3

    if input3=="D":
        input3=0
    elif input3=="M":
        input3=1
    elif input3=="S":
        input3=2
    elif input3=="W":
        input3=3
    
    continue_button2= st.button("Proceed")
    if continue_button2:
        # Make predictions using the loaded model
        prediction = model.predict([[ input2, input3, input4, input5, input6,  input7,input8, input9,input10, input11, input12, input13, input14,
                                input15, input16 ]])

        # Display the prediction
        st.subheader("Prediction")
        st.write(prediction)




