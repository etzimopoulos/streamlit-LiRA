# Angelo's first python script in Spyder
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.

st.title('LiRA - The Linear Regression App')
st.header('H1')
st.write("Hello Streamlit World")

st.write('To start with we will simulate a Linear Function which will be used to generate our data. In order to do so please select the number of samples:')

# Number of Samples
n = st.slider('Number of samples', 50, 100)
st.write("The number of data points generated will be", n)
