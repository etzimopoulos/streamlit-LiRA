#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 06:25:51 2020

@author: etzimopoulos
"""

import numpy as np
import plotly.express as px
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import time

st.title("Animation Time Trial")
mode = st.selectbox("Select Plotting Library",('Plotly','Altair','Matplotlib'))
FPS = st.slider('Attempted FPS',0,120,30)
numPoints = st.slider('Number of Plot Points',1,5000,5)
animTime = st.slider('Time to Animate',1,10,1)

#Data
X = np.random.rand(numPoints)
Y = np.random.rand(numPoints)
data = pd.DataFrame({"x":X,"y":Y})

extraTime = 0

if st.button("Animate"):

    if mode == "Plotly":
        fig1 = px.scatter(data,x="x",y="y")
        drawn_fig1 = st.plotly_chart(fig1,use_container_width = True)
        for i in range(FPS*animTime):
            frameStartTime = time.time()
            X = X +1.0/FPS
            Y = Y +1.0/FPS
            data = pd.DataFrame({"x":X,"y":Y})
            frameDrawTime = time.time()
            fig1 = px.scatter(data,x="x",y="y")
            drawn_fig1.plotly_chart(fig1,use_container_width = True)
            frameEndTime = time.time()
            extraTime = extraTime + 1/FPS - (frameEndTime-frameDrawTime)
            try:
                time.sleep(1/FPS-(frameEndTime-frameStartTime))
            except:
                time.sleep(1/FPS)

    if mode == "Altair":
        fig1 = alt.Chart(data).mark_point().encode(x="x",y="y")
        drawn_fig1 = st.altair_chart(fig1,use_container_width = True)
        for i in range(FPS*animTime):
            frameStartTime = time.time()
            X = X +1.0/FPS
            Y = Y +1.0/FPS
            data = pd.DataFrame({"x":X,"y":Y})
            frameDrawTime = time.time()
            fig1 = alt.Chart(data).mark_point().encode(x="x",y="y")
            drawn_fig1.altair_chart(fig1,use_container_width = True)
            frameEndTime = time.time()
            extraTime = extraTime + 1/FPS - (frameEndTime-frameDrawTime)
            try:
                time.sleep(1/FPS-(frameEndTime-frameStartTime))
            except:
                time.sleep(1/FPS)

    if mode == "Matplotlib":
        fig1 = plt.scatter(X,Y)
        drawn_fig1 = st.pyplot()
        for i in range(FPS*animTime):
            frameStartTime = time.time()
            X = X +1.0/FPS
            Y = Y +1.0/FPS
            data = pd.DataFrame({"x":X,"y":Y})
            frameDrawTime = time.time()
            fig1 = plt.scatter(X,Y)
            drawn_fig1.pyplot()
            frameEndTime = time.time()
            extraTime = extraTime + 1/FPS - (frameEndTime-frameDrawTime)
            try:
                time.sleep(1/FPS-(frameEndTime-frameStartTime))
            except:
                time.sleep(1/FPS)          

    extraTimePerFrame = extraTime/(FPS*animTime)

    if extraTimePerFrame > 0:
        st.success("Real time animation sucessful." +  " Extra time per frame was " + str(extraTimePerFrame))
    else:
        st.error("Real time animation failure." +  " Short time per frame was " + str(extraTimePerFrame))