from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
import pickle
from io import BytesIO
import cv2
from keras.preprocessing import image
st.set_option('deprecation.showfileUploaderEncoding', False)

classifier = load_model("classifier.h5")

STYLE = """
<style>
img {
    max-width: 50%;
}
</style>
"""

pneumonia_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> Prediction: Pneumatic Lung</h2>
       </div>
    """
    
    
normal_html="""  
      <div style="background-color:#74F61D;padding:10px >
       <h2 style="color:white;text-align:center;"> Prediction: Normal Lung </h2>
       </div>
    """


st.title("Lung X-RAY classification")
st.markdown(STYLE, unsafe_allow_html = True)

file = st.file_uploader("Upload the Lung X-RAY image to be analysed", type= ["PNG", "JPEG","JPG"])
show_file = st.empty()

if not file:
    show_file.info("Please upload a file of type: " + ", ".join(["PNG","JPEG"]))
else:    
    content = file.getvalue()
        
    if isinstance(file, BytesIO):
        show_file.image(file)
        # image = Image.open(file)
        test_image = image.load_img(file,target_size = (64,64))
        test_image = image.img_to_array(test_image)
        
        
        test_image = np.expand_dims(test_image,axis=0)
        result = classifier.predict(test_image)
        if (result[0][0]) == 1:
            st.markdown(pneumonia_html,unsafe_allow_html=True)
        else:
            st.markdown(normal_html,unsafe_allow_html=True)

    