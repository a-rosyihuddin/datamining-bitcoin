import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

df_train = pd.read_csv("data/BitcoinDataset.csv")
y = df_train['Label']

def minMax():
  st.write('Data Awal Sebelum di lakukan Preprocessing')
  st.dataframe(df_train)
  
  st.write('Data setelah dilakukan Preprocessing menggunakan Min-Max Scaler')
  scaler = MinMaxScaler()
  df_train_pre = scaler.fit_transform(df_train.drop(columns=["Label", "Date"]))
  st.dataframe(df_train_pre)
    
  # Save Scaled
  joblib.dump(df_train_pre, 'model/df_train_pre.sav')
  joblib.dump(scaler,'model/df_scaled.sav')

def label():
  return preprocessing.LabelEncoder().fit_transform(y)