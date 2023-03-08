import streamlit as st
import pickle
import pandas as pd

with open ('pipeline.pkl','rb') as f:
    pipeline= pickle.load(f)

df = pd.read_csv('reddit_posts.csv')

df= df.loc[(df.selftext.notna())&(len(df.selftext)!=0), :]

st.dataframe(df.sample(5))

test_docs= [doc for doc in df.selftext]

predictions = pipeline.predict(test_docs)

df['predictions']= predictions

df= df.loc[:, ['selftext', 'predictions']]

st.write(df)




