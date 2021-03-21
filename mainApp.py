import sentimentSnalysis
import app2
import eda
import streamlit as st

PAGES ={
    "Train Model":sentimentSnalysis,
    "EDA ": eda
}

st.sidebar.title("Navigation")

selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()