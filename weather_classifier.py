import streamlit as st
import numpy as np
import torch 
from torchvision import transforms
from classifer import SimpleCnn, int_to_label, weather_predictor
from transformer import image_transformer

st.write("""
# Weather prediction application

This app attempts to predict the weather! \n
You may take a picture with you phone and upload it
or an image from the web (.png or .Jpg formats only)\n
Please ignore the error message and proceed to upload your image :)
""")


try:
    # fetches image by prompting user to upload an image
    uploaded_file = st.file_uploader("Drop your image here",type=['png','Jpg', '.JPEG'])
except:
    print("Nothing")


# display image
st.image(uploaded_file, caption="Your image", width=224)

# transform image to desired dimension
image = image_transformer(uploaded_file)

# import the pre-trained model
model = torch.load("weather_classifier.pth", map_location=torch.device('cpu'))

# make a prediction 
output = weather_predictor(image, model)

# print output
st.write(output)

st.write("""
This prediction is made my a machine learning model
based on neural network architecture.
""")




