import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

st.title("Classifying Bookings")
def read_csv(file):
  return pd.read_csv(file)

def display_head(dataframe, n_rows = 5):
  return dataframe.head(n_rows)

file = 'C:\\Users\\muthu.g.lv\\Documents\\Case Study\\Dataset\\Classification.csv'
df = read_csv(file)
def get_numerical_column(dataframe):
  return dataframe.columns[dataframe.dtypes != "object"]

numerical_columns = get_numerical_column(df)
def get_categorical_column(dataframe):
  return dataframe.columns[dataframe.dtypes == "object"]

categorical_columns = get_categorical_column(df)

# Remove duplication
def duplicate_check(df):
    has_duplicates = df.duplicated().sum()
    #print("Total duplicates present : ", has_duplicates)
    if has_duplicates!=0:
         df = df.drop_duplicates()
    else:
         pass   
duplicate_check(df) 

# Encoding col
def encode_column(df, column):
  label = LabelEncoder()
  df[column] = label.fit_transform(df[column])

columns_to_encode = ['type_of_meal_plan', 'room_type_reserved',
       'market_segment_type', 'target']

for column in columns_to_encode:
  encode_column(df,column)

df = df[['no_of_adults', 'no_of_children', 'no_of_weekend_nights','no_of_week_nights', 'type_of_meal_plan', 'required_car_parking_space',
       'room_type_reserved', 'lead_time', 'arrival_month',
       'market_segment_type', 'repeated_guest',
       'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
       'avg_price_per_room', 'no_of_special_requests', 'target']]


cols=df.columns
ms=MinMaxScaler()
df=ms.fit_transform(df)

df=pd.DataFrame(df,columns=cols)
y = df['target']
X = df.drop(['target'], axis = 1)

smote = SMOTE()
X,y = smote.fit_resample(X,y)

rfc = RandomForestClassifier(max_depth= None, min_samples_leaf= 1, min_samples_split= 5, n_estimators= 200)
rfc.fit(X, y)

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
        df = df[['no_of_adults', 'no_of_children', 'no_of_weekend_nights',
        'no_of_week_nights', 'type_of_meal_plan', 'required_car_parking_space',
        'room_type_reserved', 'lead_time', 'arrival_month',
        'market_segment_type', 'repeated_guest',
        'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled','avg_price_per_room', 'no_of_special_requests']]      

        #getting numerical columns
        def get_numerical_column(dataframe):
            return dataframe.columns[dataframe.dtypes != "object"]

        numerical_columns = get_numerical_column(df)
    
        def get_categorical_column(dataframe):
            return dataframe.columns[dataframe.dtypes == "object"]

        categorical_columns = get_categorical_column(df)

        def missing_check(dataframe):
            return dataframe.isnull().sum().any()
        missing_check(df)

        def fill_missing_values_mean(columns_to_fill, dataframe):
            for col_name in columns_to_fill:
                mean = dataframe[col_name].mean()
                dataframe[col_name].fillna(mean,inplace=True)

    
        def duplicate_check(df):
            has_duplicates = df.duplicated().sum()
            #print("Total duplicates present : ", has_duplicates)
            if has_duplicates!=0:
                df = df.drop_duplicates()
            else:
                pass   
        duplicate_check(df) 
    

   
        continue_button =st.button("Proceed")
        if continue_button:
            columns_to_encode = categorical_columns
            def encode_column(df, column):
                for i in column:
                    label = LabelEncoder()
                    df[i] = label.fit_transform(df[i])
            
            encode_column(df,columns_to_encode)
                #########################
            st.subheader("Predicted Data")
            #st.write(df.head(5))
            y_pred = rfc.predict(df)
            # Using DataFrame.insert() to add a column
            #df.insert("Predicted value", y_pred, True)

            df2 = df_copy.assign(Predict=y_pred)
            df2['Predicted'] = df2['Predict'].apply(lambda x: 'Canceled' if x == 0 else 'Not_Canceled')
            df2 = df2.drop(['Predict'], axis = 1)

            st.write(df2.head(5))

import streamlit as st



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

