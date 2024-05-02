from fastai.vision.widgets import *
from fastai.vision.all import *
import os
import cv2 as cv2

import pathlib
temp = pathlib.PosixPath
pathlib.WindowsPath =  pathlib.PosixPath
import streamlit as st



# method to resize the bulk files
def trans_image(filename):
        
    #LOAD/OPEN IMAGE
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Format Image
    new_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
    new_img = cv2.bitwise_not(new_img)

    print('Resized and saved {} successfully.'.format(filename))
    return new_img






class Predict:
    def __init__(self, filename):
        self.learn_inference = load_learner(filename)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
            self.get_prediction()
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(180,180), caption='Uploaded Image')

    def get_prediction(self):

        if st.button('Classify'):
            pred, pred_idx, probs = self.learn_inference.predict(self.img)
            st.write(f'Prediction: {pred};  Confidence: {probs[pred_idx]:.04f}')
        else: 
            st.write(f'Click the button to classify') 

if __name__=='__main__':

    # file_name=('firstPass_128.pkl')

    akan_name =('AkanfirstPass.pkl')

    # predictor = Predict(file_name)
    
    predictor = Predict(akan_name)

# use this command to run webapp
    # streamlit run predict_image.py