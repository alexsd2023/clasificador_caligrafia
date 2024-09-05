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
from math import ceil

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
    
    #st.image(file)
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
        #st.write('Prediction: ', prediction[0])
        return(prediction[0]) 
    
def update(df, image, col):
    #df.at[image, col] = st.session_state[f'{col}_{image}']
    #print('IMAGEN MODIFICADA: ', image)
    df.at[0, 'Incorrect'] = True
    st.session_state.df= df

def run():    
   
    
    st.title("Classifier")
    model = tf.saved_model.load('./models/')
    classes = [ "ITALICA_REDONDA" ,  "ITALICA_CURSIVA" ,  "PROCESAL_2" ,  "Procesal_encadenada" , ]
    
    
    #modelname = st.radio(
    #     "Select a model",
    #    ["EfficientNET", "SVN","MobileNET", "VGG 16/19"],
    #     index=0,
    #)
    #st.set_option('widemode', True)
    uploaded_files= st.file_uploader("Choose the images files", type={'jpg'},  accept_multiple_files= True)
    if uploaded_files != None:
        df= pd.DataFrame({'Image Name': [file.name for file in uploaded_files],'Label': ['Unlabelled']*len(uploaded_files), 'Incorrect': [False]*len(uploaded_files)})
        st.session_state.df= df
    else:
        df= pd.DataFrame()
    

    controls= st.columns(3)
    with controls[0]:
         batch_size= st.select_slider("Batch size:", range(5, 50, 5), key='slider_batch')
    with controls[1]:
        row_size= st.select_slider("Row size:", range(1, 6), value= 5, key='slider_row_size')
    num_batches= ceil(len(uploaded_files)/batch_size)
    with controls[2]:
        page= st.selectbox('Page', range(1, num_batches+1))

    if len(uploaded_files) != 0:    
        batch= uploaded_files[(page-1)*batch_size:page*batch_size]
        grid= st.columns(row_size)
        col= 0
        for image in batch:
            with grid[col]:
                result= predict(image, 'SVN', classes, model)
                caption= image.name + '\n' + str(result)
                st.image(image, width= 150, caption=caption)
                # value= df.at[image.name, 'Label'],
                st.checkbox('Incorrect', key=f'label_{image.name}', value= False,  on_change= update, args=(df, image.name, 'Label'))
            col= (col+1) % row_size

        st.write('Predictions')   
        
        if 'df' in st.session_state.keys():
            df= st.session_state.df
        st.dataframe(df[df['Incorrect'] == False])
   

