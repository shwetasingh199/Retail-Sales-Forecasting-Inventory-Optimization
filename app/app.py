import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import os

# fix import path
sys.path.append(os.path.dirname(__file__))

from utils import inventory_calc, eoq_calc

# load data
df = pd.read_csv("../data/processed/feature_data.csv")

# load model
with open("../models/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📊 Retail Forecasting & Inventory Optimization")

# selection
store = st.selectbox("Select Store", df['Store'].unique())
sku = st.selectbox("Select SKU", df['SKU'].unique())

filtered = df[(df['Store']==store) & (df['SKU']==sku)]

st.subheader("📈 Sales Trend")
st.line_chart(filtered['Sales'])

# prediction
features = [
    'lag_1','lag_7','lag_14',
    'rolling_mean_7','rolling_mean_14',
    'weekday'
]

X = filtered[features].tail(1)
pred = model.predict(X)[0]

st.subheader("🔮 Forecast")
st.write("Next Day Forecast:", round(pred,2))

# inventory
lead_time = st.slider("Lead Time (days)", 1, 30, 7)

ss, rop = inventory_calc(filtered['Sales'], lead_time)
eoq = eoq_calc(filtered['Sales'].mean())

st.subheader("📦 Inventory Recommendation")
st.write("Safety Stock:", ss)
st.write("Reorder Point:", rop)
st.write("EOQ:", eoq)

# reorder alert
current_stock = st.number_input("Current Stock", value=500)

if current_stock < rop:
    st.error("⚠️ Reorder Required!")
else:
    st.success("✅ Stock is sufficient")

# export PO
if st.button("Generate Purchase Order"):
    po = pd.DataFrame({
        "Store":[store],
        "SKU":[sku],
        "Order_Qty":[eoq]
    })
    
    po.to_csv("../outputs/purchase_orders.csv", index=False)
    
    st.success("📄 Purchase Order Generated!")