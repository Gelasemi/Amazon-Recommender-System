import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, KNNBasic, accuracy
from surprise.model_selection import train_test_split
import zipfile
import os
import gzip
import requests
from io import BytesIO

# Title of the app
st.title("Amazon Product Recommendation System")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Choose a section:", [
    "Objectives & Goals", "Upload Data", "Algorithm Selection", "Results", "Interpretations", "Plots", "Decision"])

# Function to download the df_final CSV from Google Drive
def download_file_from_google_drive(url):
    file_id = url.split('/d/')[1].split('/')[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(download_url)
    if response.status_code == 200:
        return pd.read_csv(BytesIO(response.content))
    else:
        st.error("Failed to download the file.")
        return None

# Section 1: Objectives & Goals
if options == "Objectives & Goals":
    st.header("Project Objectives & Goals")
    st.write("""
    **Objective:** Build a recommendation system to recommend products to customers based on their previous ratings for other products.
    
    **Goals:**
    - Extract meaningful insights from the data.
    - Build a recommendation system that helps in recommending products to online consumers.
    - Evaluate the performance of different recommendation algorithms.
    """)

# Section 2: Upload Data
elif options == "Upload Data":
    st.header("Upload Dataset")
    
    # Handle the file upload with support for large files
    uploaded_file = st.file_uploader("Upload your dataset (CSV, CSV.GZ, or ZIP format)", type=["csv", "csv.gz", "zip"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall("temp_zip_extract")
                extracted_files = zip_ref.namelist()
                csv_file = next((f for f in extracted_files if f.endswith('.csv')), None)
                if csv_file:
                    # Using a generator to load the file in chunks
                    def load_large_csv(file_path):
                        chunk_size = 100000  # Adjust chunk size to suit your memory limits
                        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                            yield chunk

                    df = pd.concat(load_large_csv(os.path.join("temp_zip_extract", csv_file)), ignore_index=True)
                    st.write("Dataset Preview:")
                    st.write(df.head())
                    st.session_state['df'] = df
                else:
                    st.error("No CSV file found in the ZIP archive.")
                for file in extracted_files:
                    os.remove(os.path.join("temp_zip_extract", file))
                os.rmdir("temp_zip_extract")
        elif uploaded_file.name.endswith('.gz'):
            with gzip.open(uploaded_file, 'rb') as f_in:
                # Read the gzipped file in chunks
                def load_large_csv(file_obj):
                    chunk_size = 100000  # Adjust chunk size to suit your memory limits
                    for chunk in pd.read_csv(file_obj, chunksize=chunk_size):
                        yield chunk

                df = pd.concat(load_large_csv(f_in), ignore_index=True)
                st.write("Dataset Preview:")
                st.write(df.head())
                st.session_state['df'] = df
        elif uploaded_file.name.endswith('.csv'):
            # Handle CSV directly if it's not too large
            def load_large_csv(file_path):
                chunk_size = 100000  # Adjust chunk size to suit your memory limits
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    yield chunk

            df = pd.concat(load_large_csv(uploaded_file), ignore_index=True)
            st.write("Dataset Preview:")
            st.write(df.head())
            st.session_state['df'] = df
        else:
            st.error("Unsupported file format. Please upload a CSV, CSV.GZ, or ZIP file.")
    
    # Alternatively, download df_final from Google Drive
    st.write("Alternatively, you can download the final dataset (df_final) from Google Drive.")
    url = "https://colab.research.google.com/drive/1EFq8zhqsPTh1npZct6frYvNl2hzCdoBo?usp=drive_link"
    if st.button("Download df_final"):
        df_final = download_file_from_google_drive(url)
        if df_final is not None:
            st.write("df_final Dataset Preview:")
            st.write(df_final.head())
            st.session_state['df'] = df_final

# Section 3: Algorithm Selection
elif options == "Algorithm Selection":
    st.header("Select Recommendation Algorithm")
    if 'df' not in st.session_state:
        st.error("Please upload a dataset first.")
    else:
        df = st.session_state['df']
        algorithm = st.selectbox("Choose an algorithm:", [
            "Rank-Based", "Collaborative Filtering (User-User)", "Collaborative Filtering (Item-Item)"])

        if algorithm == "Rank-Based":
            st.write("Rank-Based Recommendation System Selected")
            average_rating = df.groupby('prod_id')['rating'].mean()
            count_rating = df.groupby('prod_id')['rating'].count()
            final_rating = pd.DataFrame({'average_rating': average_rating, 'count_rating': count_rating})
            final_rating = final_rating.sort_values('average_rating', ascending=False)
            st.write("Top 5 Products by Average Rating:")
            st.write(final_rating.head())
            st.session_state['result'] = final_rating.head()

        elif algorithm == "Collaborative Filtering (User-User)":
            st.write("Collaborative Filtering (User-User) Selected")
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(df[['user_id', 'prod_id', 'rating']], reader)
            trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
            sim_options = {'name': 'cosine', 'user_based': True}
            model = KNNBasic(sim_options=sim_options)
            model.fit(trainset)
            predictions = model.test(testset)
            rmse = accuracy.rmse(predictions)
            st.write("RMSE for Collaborative Filtering (User-User):", rmse)
            st.session_state['result'] = predictions

# Section 4: Results
elif options == "Results":
    st.header("Algorithm Results")
    if 'result' not in st.session_state:
        st.error("No results to display. Please run an algorithm first.")
    else:
        st.write(st.session_state['result'])

# Section 5: Interpretations
elif options == "Interpretations":
    st.header("Interpretations")
    if 'result' not in st.session_state:
        st.error("No interpretations available. Please run an algorithm first.")
    else:
        if isinstance(st.session_state['result'], pd.DataFrame):
            st.write("Rank-Based Recommendations: Top 5 products by average rating.")
        else:
            st.write("Collaborative Filtering: Predictions made using User-User similarity with RMSE evaluation.")

# Section 6: Plots
elif options == "Plots":
    st.header("Visualizations")
    if 'df' not in st.session_state:
        st.error("No data available for visualization. Please upload a dataset first.")
    else:
        df = st.session_state['df']
        st.write("Rating Distribution:")
        fig, ax = plt.subplots()
        df['rating'].value_counts().sort_index().plot(kind='bar', ax=ax)
        st.pyplot(fig)

# Section 7: Decision
elif options == "Decision":
    st.header("Final Decision")
    if 'result' not in st.session_state:
        st.error("No decision can be made yet. Please run an algorithm first.")
    else:
        if isinstance(st.session_state['result'], pd.DataFrame):
            st.write("Implement the top 5 products in the recommendation system.")
        else:
            st.write("Use Collaborative Filtering for personalized recommendations.")
