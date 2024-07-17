import streamlit as st
import streamlit.components.v1 as components


st.title("Graph representations of the data")


# add interactive graph element
html = open("/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Bongard/HotIn/Vis/0.html",'r').read()

components.html(html, height=400, width=400)