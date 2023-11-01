import pandas as pd
import numpy as np
import streamlit as st
import re
import pickle
import statistics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df_1=pd.read_csv('ResaleFlatPricesBasedonApprovalDate19901999.csv')
df_2=pd.read_csv('ResaleFlatPricesBasedonApprovalDate2000Feb2012.csv')
df_3=pd.read_csv('ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv')
df_4=pd.read_csv('ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv')
df_5=pd.read_csv('ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv')
df=pd.concat([df_1,df_2,df_3,df_4,df_5],axis=0,ignore_index=True)


l=list(df['month'])
b=[]
a=[i.split('-') for i in l]
for i in a:
    b.append(i[1])

l=list(df['month'])
year=[]
a=[i.split('-') for i in l]
for i in a:
    year.append(i[0])

df['Month']=b
df['Year']=year


def get_median(x):
    split_list=x.split('TO')
    flo_list=[float(i) for i in split_list]
    median=statistics.median(flo_list)
    return median
df['storey_median']=df['storey_range'].apply(get_median)

df.drop(columns=['remaining_lease','street_name','month','town','storey_range'],axis=1,inplace=True)

df['tr_floor_area_sqm']=np.log(df['floor_area_sqm'])
df['tr_lease_commence_date']=np.log(df['lease_commence_date'])
df['tr_resale_price']=np.log(df['resale_price'])
df['tr_storey_median']=np.log(df['storey_median'])


x=df[['flat_type', 'tr_floor_area_sqm', 'flat_model',
       'tr_lease_commence_date', 'Month', 'Year',
       'tr_storey_median']]
y=df['tr_resale_price']

ohe_1 = OneHotEncoder(handle_unknown='ignore')
x_ohe_1 = ohe_1.fit_transform(x[['flat_type']]).toarray()

ohe_2=OneHotEncoder(handle_unknown='ignore')
x_ohe_2=ohe_2.fit_transform(x[['flat_model']]).toarray()

ohe_3=OneHotEncoder(handle_unknown='ignore')
x_ohe_3=ohe_3.fit_transform(x[['Month']]).toarray()

ohe_4=OneHotEncoder(handle_unknown='ignore')
x_ohe_4=ohe_4.fit_transform(x[['Year']]).toarray()

X = np.concatenate((x[['tr_floor_area_sqm','tr_lease_commence_date','tr_storey_median']].values, x_ohe_1,x_ohe_2,x_ohe_3,x_ohe_4), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

model=RandomForestRegressor()
model.fit(X_train,y_train)
pred=model.predict(X_test)

r_2=r2_score(y_test, pred)

mse=mean_squared_error(y_test, pred)

mae=mean_absolute_error(y_test, pred)

with open('regressor.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('encoder_1.pkl','wb') as file:
    pickle.dump(x_ohe_1,file)
with open('encoder_2.pkl','wb') as file:
    pickle.dump(x_ohe_2,file)

with open('encoder_3.pkl','wb') as file:
    pickle.dump(x_ohe_3,file)

with open('encoder_4.pkl','wb') as file:
    pickle.dump(x_ohe_4,file)


st.set_page_config(layout="wide")

st.title("RESALE PRICING")

tab=st.tabs(['Predicting resale price'])

flat_type_select=['1 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE',
   'MULTI GENERATION', 'MULTI-GENERATION']
flat_model_select=['IMPROVED', 'NEW GENERATION', 'MODEL A', 'STANDARD', 'SIMPLIFIED',
   'MODEL A-MAISONETTE', 'APARTMENT', 'MAISONETTE', 'TERRACE',
   '2-ROOM', 'IMPROVED-MAISONETTE', 'MULTI GENERATION',
   'PREMIUM APARTMENT', 'Improved', 'New Generation', 'Model A',
   'Standard', 'Apartment', 'Simplified', 'Model A-Maisonette',
   'Maisonette', 'Multi Generation', 'Adjoined flat',
   'Premium Apartment', 'Terrace', 'Improved-Maisonette',
   'Premium Maisonette', '2-room', 'Model A2', 'Type S1', 'Type S2',
   'DBSS', 'Premium Apartment Loft', '3Gen']
month_select=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11','12']
year_select=['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997',
   '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005',
   '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2015',
   '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023',
   '2013', '2014']
with st.form("my_form"):
    col1, col2 = st.columns([3,3])
    with col1:
        st.write(' ')
        Flat_type = st.selectbox("Flat_type", flat_type_select, key=1)
        Flat_model = st.selectbox("Flat_model",flat_model_select , key=2)
        Month = st.selectbox("Month",month_select , key=3)
        Year= st.selectbox("Year", sorted(year_select), key=4)

    with col2:
        floor_area=st.text_input("Enter Floor Area (Min:28.0 & Max:307.0)")
        lease_commence=st.text_input("Enter Lease Commence Date(Min:1966 & Max:2022)")
        storey_median=st.text_input("Enter Storey(Min:2.0 & Max:50.0)")

        submit = st.form_submit_button(label="PREDICT RESALE PRICE")

    flag = 0
    pattern = '[0-9]*\.?[0-9]+'
    for i in [floor_area, lease_commence, storey_median]:
        if re.match(pattern, i):
            pass
        else:
            flag = 1
            break

if submit and flag == 1:
    if len(i) == 0:
        st.write("please enter a valid number space not allowed")
    else:
        st.write("You have entered an invalid value: ", i)

if submit and flag == 0:


    with open(r"regressor.pkl", 'rb') as file:
        loaded_model = pickle.load(file)

    with open(r"encoder_1.pkl", 'rb') as f:
        oh_1_load = pickle.load(f)

    with open(r"encoder_2.pkl", 'rb') as f:
        oh_2_load = pickle.load(f)

    with open(r"encoder_3.pkl", 'rb') as f:
        oh_3_load = pickle.load(f)

    with open(r"encoder_4.pkl", 'rb') as f:
        oh_4_load = pickle.load(f)

    new_sample = np.array([[np.log(float(floor_area)),np.log(float(lease_commence)),float(storey_median),
                            Flat_type,Flat_model,Month,Year]])
    new_sample_ohe_1 = oh_1_load.transform(new_sample[:, [3]]).toarray()
    new_sample_ohe_2 = oh_2_load.transform(new_sample[:, [4]]).toarray()
    new_sample_ohe_3=oh_3_load.transform(new_sample[:, [5]]).toarray()
    new_sample_ohe_4=oh_4_load.transform(new_sample[:, [6]]).toarray()
    new_sample = np.concatenate((new_sample[:, [0, 1, 2]], new_sample_ohe_1, new_sample_ohe_2,new_sample_ohe_3,new_sample_ohe_4), axis=1)
    new_pred = loaded_model.predict(new_sample)[0]
    st.write('## :green[Predicted Resale Price:] ', round(np.exp(new_pred)))




