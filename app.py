import streamlit as st
from PIL import Image
import cv2
import numpy as np
import time
from predict import *
st.title('Segmentation')
def reset(): 
    st.session_state.image = None
uploaded_file = st.file_uploader("Choose a file", accept_multiple_files= False, on_change= reset)
bytes_data  = None
segmented_image = None
image_holder = st.empty()
def handleClick():
    if  uploaded_file is not None:
        send = True
        image = Image.open(uploaded_file)
        image = np.array(image)
        image_holder.empty()
        st.session_state.image = predict(image)
if uploaded_file is not None:
        try:
            if not isinstance(st.session_state.image, np.ndarray):
                st.session_state.image = uploaded_file.read()
        except:
                st.session_state.image = uploaded_file.read()
        image = image_holder.image(st.session_state.image)
        st.button("Segmentation", disabled = False, type="primary", on_click=handleClick)
else:
    st.button("Segmentation", disabled = True, type="primary", on_click=handleClick)

