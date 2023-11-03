import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from PIL import Image
import numpy as np

st.title("Crop Image Classification")
selected_opt = st.selectbox('Select a way to use image', options=[
    'Upload from directory', 'Download from URL'])

img, downloaded = False, False
if selected_opt == 'Upload from directory':
    img = st.file_uploader('Select an image')
else:
    img_url = st.text_input("Enter image url")
    img_name = 'img_downloaded.jpg'

    download_img_btn = st.button("Download")
    if download_img_btn:
        try:
            urlretrieve(img_url, img_name)
            img_name = img_name
        except Exception as e:
            st.error(e)
    downloaded = True

if img or downloaded:
    try:
        if selected_opt == 'Upload from directory':
            image = Image.open(img)
        else:
            image = Image.open(img_name)
    except Exception as e:
        st.error('Unsupported file format !')
    try:
        image = np.array(image)
        image_tensor = tf.image.resize(image, size=(224, 224))
        st.image(image, width=224)
    except Exception as e:
        st.error(e)


@st.cache_resource
def load_my_model():
    try:
        model = tf.keras.models.load_model('./effnetb2_model.keras')
    except Exception as e:
        st.write(e)
    return model


model = load_my_model()
predict_btn = st.button('Predict')

if predict_btn:
    def predict_crop_img(img):
        crops = {'jute': 0, 'maize': 1, 'rice': 2, 'sugarcane': 3, 'wheat': 4}
        img = tf.expand_dims(img, axis=0)
        prediction = model.predict(img, verbose=0)
        crop_ind = np.argmax(prediction)
        pred_str = f"### Predicted crop : `{list(crops)[crop_ind]}`"
        st.write(pred_str)
    try:
        predict_crop_img(image_tensor)
    except Exception as e:
        st.write(e)
