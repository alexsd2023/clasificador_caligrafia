#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from PIL import Image, ImageOps
import glob
import os

from annotated_text import annotated_text
from st_click_detector import click_detector
from bs4 import BeautifulSoup
from utils import annotate_txt
import matplotlib.colors as mcolors

import tensorflow as tf
import numpy as np
import io

import joblib
import cv2  
from skimage import feature

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
            self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist
    

def predict(file, modelname, classes, model):
    
    st.image(file)
    if modelname == 'EfficientNET':
        img= Image.open(file).convert('RGB')
        img= img.convert('RGB')  
        img = img.resize((300, 300 * img.size[1] // img.size[0]))
        inp_numpy = np.array(img)[None]
            
        inp = tf.constant(inp_numpy, dtype='float32')
        class_scores = model(inp)[0].numpy()
        
        st.write("Class predicted : ", classes[class_scores.argmax()])
        data= []
        for index in range(0, len(class_scores)):
            data.append([classes[index], round(class_scores[index], 2)])
        df= pd.DataFrame(data, columns=('Class', 'Score'))
        st.table(df) 

    elif modelname == 'SVN':
        desc = LocalBinaryPatterns(24, 8)
        
        model = joblib.load("./models/caligrafia.pkl")

        # load the image, convert it to grayscale, describe it,
        # and classify it
        #img= Image.open(file).convert('RGB')
        #gray= ImageOps.grayscale(img)

        image= Image.open(file)
        img_array= np.array(image)
        image= cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        hist = desc.describe(gray)
        prediction = model.predict(hist.reshape(1, -1))
        st.write('Prediction: ', prediction[0])
         
def run():    
   
    
    st.title("Classifier")
    model = tf.saved_model.load('./models/')
    classes = [ "ITALICA_REDONDA" ,  "ITALICA_CURSIVA" ,  "PROCESAL_2" ,  "Procesal_encadenada" , ]
    #st.write(tf.__version__)
    modelname = st.radio(
         "Select a model",
        ["EfficientNET", "SVN","MobileNET", "VGG 16/19"],
         index=0,
    )
    st.session_state['modelname']= modelname
    #st.set_option('widemode', True)
    uploaded_files= st.file_uploader("Choose the images files", type={'jpg'},  accept_multiple_files= True)
    for uploaded_file in uploaded_files:
        bytes_data= uploaded_file.read()
        st.write("Predictions with model: ", modelname)
        st.write("filename: ", uploaded_file.name)
        if modelname == 'SVN':
            model= None
        predict(uploaded_file, modelname, classes, model)
        #st.write(bytes_data) 
    
   

