#Streamlit App for Shopper Spectrum
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved models
kmeans = pickle.load(open('rfm_kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('rfm_scaler.pkl', 'rb'))
sim_matrix = pd.read_pickle('product_similarity_matrix.pkl')

# App Title
st.title("🛍️ Shopper Spectrum: Product Recommendation & Segmentation")
st.markdown("""
This app helps you:
- 📊 Predict Customer Segments based on RFM values
- 🛒 Recommend Similar Products using Collaborative Filtering
""")

# Sidebar tab-like navigation
selected_tab = st.sidebar.radio("📌 Home", ["Product Recommendation", "Customer Segmentation"])

# 📌 Product Recommendation
if selected_tab == "Product Recommendation":
    st.header("📌 Product Recommendation")
    st.subheader("🔍 Find Similar Products")
    product_input = st.text_input("Enter Product Name:", "WHITE HANGING HEART T-LIGHT HOLDER")
    if st.button("🔎 Get Recommendations"):
        if product_input in sim_matrix:
            recs = sim_matrix[product_input].sort_values(ascending=False)[1:6]
            st.success("Top 5 similar products:")
            for i, (item, score) in enumerate(recs.items(), 1):
                st.write(f"{i}. {item} (Similarity Score: {score:.2f})")
        else:
            st.error("Product not found. Please check spelling or try another name.")

# 👤 Customer Segmentation
elif selected_tab == "Customer Segmentation":
    st.header("👤 Customer Segmentation")
    st.subheader("📈 Predict Customer Segment")
    recency = st.number_input("Recency (days since last purchase):", min_value=0, max_value=500, value=30)
    frequency = st.number_input("Frequency (total purchases):", min_value=1, max_value=100, value=5)
    monetary = st.number_input("Monetary (total amount spent):", min_value=1, value=500)

    if st.button("📌 Predict Segment"):
        rfm_input = np.array([[recency, frequency, monetary]])
        rfm_scaled = scaler.transform(rfm_input)
        cluster_label = kmeans.predict(rfm_scaled)[0]

        # Rule-based labeling for clarity
        if recency <= 50 and frequency >= 5 and monetary >= 1000:
            segment = 'High-Value'
        elif recency > 100 and frequency <= 2 and monetary < 500:
            segment = 'At-Risk'
        elif frequency >= 3 and monetary >= 500:
            segment = 'Regular'
        else:
            segment = 'Occasional'

        st.success(f"Predicted Cluster: {cluster_label} → Segment: **{segment}**")
