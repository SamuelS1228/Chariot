
import streamlit as st
import pandas as pd
from optimization import optimize
from visualization import plot_network, summary

st.set_page_config(page_title="Test Tool", layout="wide")
st.title("Warehouse Tool")

upload = st.file_uploader("Store CSV")
if upload:
    df = pd.read_csv(upload)
    if 'DemandLbs' not in df.columns:
        st.error("CSV must include DemandLbs column")
    else:
        result = optimize(
            df,
            k_vals=[3],
            rate_out_min=0.02,
            sqft_per_lb=0.02,
            cost_sqft=6.0,
            fixed_cost=250000.0
        )
        st.write(result['total_cost'])
