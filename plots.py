import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sea 
import streamlit as st 

def app (car_df):
	st.sidebar.title("Choose method of data visualisation:")
	method=st.multiselect ("Histogram", "Boxplot", "Correlation Heatmap")
	if "Histogram" in method:
		st.subheader("Histogram")
		col=st.sidebar.selectbox("Select a column to create the histogram of:", ('carwidth', 'enginesize', 'horsepower', 'drivewheel_fwd', 'car_company_buick'))
        plt.figure(figsize=(16,4))
        plt.title(f"Histogram of {col}")
		plt.hist(car_df[col],bins='sturges', edgecolor='black')
		st.pyplot()
		if "Boxplot" in method:
		st.subheader("Boxplot")
		col=st.sidebar.selectbox("Select a column to create the boxplot of:", ('carwidth', 'enginesize', 'horsepower', 'drivewheel_fwd', 'car_company_buick'))
        plt.figure(figsize=(16,4))
        plt.title(f"Boxplot of {col}")
		sea.boxplot(car_df[col])
		st.pyplot()
		if "Correlation Heatmap" in method:
		st.subheader("Correlation Heatmap")
		plt.figure(figsize=(8,8), dpi=96)
        plt.title("Correlation Heatmap")
		ax = sea.heatmap(car_df.corr(), annot = True)
	 	bottom, top = ax.get_ylim()
    	ax.set_ylim(bottom + 0.5, top - 0.5)
    	st.pyplot()
