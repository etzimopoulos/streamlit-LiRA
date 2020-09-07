#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 06:09:26 2020

@author: etzimopoulos
"""

import pandas as pd
import altair as alt
import streamlit as st

df1 = pd.DataFrame({
    'times': [1, 2, 3],
    'values': [1, 5, 4],
})

df2 = pd.DataFrame({
    'times': [2, 3, 4],
    'values': [4, 2, 3],
})

chart1 = alt.Chart(df1).mark_line().encode(x='times', y='values')
chart2 = alt.Chart(df2).mark_point().encode(x='times', y='values')

#chart1 + chart2

st.altair_chart(chart1+chart2)