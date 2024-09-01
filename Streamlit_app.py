import streamlit as st
import tensorflow as tf
import pickle as pkl
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import io

# Title of the application
st.write("Anomaly Detection Application")

# File uploader for user to upload their data file
file_upload = st.file_uploader(label="Upload your file", type=["csv", "xlsx", "xls"])

if file_upload is not None:
    file = file_upload.name
    # Read the uploaded file based on its type
    if file.endswith(".csv"):
        data = pd.read_csv(file_upload)
    elif file.endswith((".xlsx", ".xls")):
        data = pd.read_excel(file_upload)
    else:
        st.write("Unsupported file type")
    
    st.write(data.head(10))

    # Ensure 'Amount' and 'Time' columns exist before scaling
    if "Amount" in data.columns and "Time" in data.columns:
        # Scale the 'Amount' column using RobustScaler
        data["Amount"] = RobustScaler().fit_transform(data["Amount"].values.reshape(-1, 1))
        # Scale the 'Time' column using MinMaxScaler
        data["Time"] = MinMaxScaler().fit_transform(data["Time"].values.reshape(-1, 1))

        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data.select_dtypes(include=[float, int]))  # PCA only on numerical features

        # Add PCA components to the data
        data["PCA1"] = data_pca[:, 0]
        data["PCA2"] = data_pca[:, 1]
        st.write(data.shape)

        # Load pre-trained models for anomaly detection
        with open("iso_forest_model.pkl", "rb") as f:
            iso_forest_model = pkl.load(f)  # Isolation Forest model
        from tensorflow.keras.losses import BinaryCrossentropy

        custom_objects = {'BinaryCrossentropy': BinaryCrossentropy()}
        autoencoder_model = load_model("Autoencoder_model.h5", custom_objects=custom_objects)
        # autoencoder_model = tf.keras.models.load_model("Autoencoder_model.h5")  # Autoencoder model

        # Button to trigger predictions
        if st.button("Predict"):
            # Make predictions using Isolation Forest
            iso_forest_model_preds = iso_forest_model.predict(data.select_dtypes(include=[float, int]))
            # Make predictions using the autoencoder
            autoencoder_model_preds = autoencoder_model.predict(data.select_dtypes(include=[float, int]))
            autoencoder_model_preds = tf.squeeze(tf.round(autoencoder_model_preds))
            # Add predictions to the data
            data["Isolation_Forest_preds"] = iso_forest_model_preds
            data["Autoencoder_preds"] = autoencoder_model_preds
            st.write(data.head(10))

    st.write("To view the whole file, please download the file")

    # User selects the file type for download
    file_type = st.radio("Select file type: ", ("CSV", "Excel", "Json"))
    # Default file name based on selected type
    default_file_name = f"processed_data.{file_type.lower()}"
    # Optional custom file name input
    custom_file_name = st.text_input("Enter a custom file name (optional):", default_file_name)

    # Prepare data for download based on selected file type
    if file_type == "CSV":
        file_data = data.to_csv(index=False)
        mime_type = "text/csv"
    elif file_type == "Excel":
        buffer = io.BytesIO()
        data.to_excel(buffer, index=False, engine='openpyxl')
        buffer.seek(0)
        file_data = buffer.getvalue()
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif file_type == "Json":
        file_data = data.to_json(index=False)
        mime_type = "application/json"

    # Download button for the processed data
    st.download_button(label="Download", data=file_data, file_name=custom_file_name, mime=mime_type)
