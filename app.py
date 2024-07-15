import streamlit as st
import numpy as np
from PIL import Image
from utils import predict,process_image,load_save_model
# import tensorflow as tf
# print(tf.__version__)

# Set the title
st.title("Plant Leaf Detection")

# # Add a image uploader

uploaded_file=st.file_uploader(label="Upload Plant Images Only")
if uploaded_file is not None:
    # image = Image.open(uploaded_file)
    st.image(uploaded_file)
    # Load the model for prediction
    image_array=process_image(img= uploaded_file)
    
    st.write(np.shape(image_array))
    model=load_save_model()

    