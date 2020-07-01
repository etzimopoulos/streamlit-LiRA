#!/usr/bin/env python
# coding: utf-8

# # Import Libraries
import streamlit as st

import numpy as np
import pandas as pd
from numpy import array
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy import stats

st.title('LiRA - The Linear Regression App')
st.header('H1')
# # A. Solve using Analytical Calculus - Random data points

# ## Create random X and y samples


# Generate 'random' data
#np.random.seed(1)
st.write('To start with we will simulate a Linear Function which will be used to generate our data. In order to do so please select the number of samples:')

# Number of Samples
n = st.slider('Number of samples', 50, 100)
st.write("The number of data points generated will be", n)

# Create r and r1, random vectors of 100 numbers each with mean = 0 and standard deviation = 1
r = np.random.randn(n)
r1 = np.random.randn(n)

# Create random Input vector X using r
# mean = 3
# stddev = 2
X = 3 * r + 2

st.header('Residual')
# Create random Residual term Res using r
# mean = 0
stddev = st.slider ('Please enter Standard Deviation (0.0-1.0 values)',0.0,1.1)
res = stddev* r1 

# Generate Y values based on the simulated regression line and error/noise
# Population Regression Line
yreg = 2.5 + 0.35 * X 
# Adding noise/error
y = yreg + res                  

# Storing Population Regression Line "RegL", data points X and y in a data frame
rl = pd.DataFrame(
    {'X': X,
     'y': y,
     'RegL':yreg}
)

# Show the first five rows of our dataframe
rl.head()


# ## Calculate coefficients alpha and beta

# Assuming y = aX + b
# a ~ alpha
# b ~ beta

# Calculate the mean of X and y
xmean = np.mean(X)
ymean = np.mean(y)

# Calculate the terms needed for the numator and denominator of alpha
rl['CovXY'] = (rl['X'] - xmean) * (rl['y'] - ymean)
rl['VarX'] = (rl['X'] - xmean)**2

# Calculate alpha
# Numerator: Covariance between X and y
# Denominator: Variance of X
alpha = rl['CovXY'].sum() / rl['VarX'].sum()

# Calculate beta
beta = ymean - (alpha * xmean)
print('alpha =', alpha)
print('beta =',beta)


# ## Prediction - Least Squares Line
ypred = alpha * X + beta


# ## Calculate Model Metrics - RSS, RSE(σ), TSS and R^2 Statistic

# Residual Errors
RE = (rl['y'] - ypred)**2
#Residual Sum Squares
RSS = RE.sum()
print("Residual Sum of Squares (RSS) is:",RSS)

# Estimated Standard Variation (sigma) or RSE
RSE = np.sqrt(RSS/(n-2))
print("\nResidual Standar Error (Standard Deviation σ) is:",RSE)

# Total Sum of squares (TSS)
TE = (rl['y'] - ymean)**2
# Total Sum Squares
TSS = TE.sum()
print("\nTotal Sum of Squares (TSS) is:",TSS)

# R^2 Statistic
R2 = 1 - RSS/TSS
print("\n R2 Statistic is:",R2)


# ## Assessing Coefficients accuracy


# Degrees of freedom
df = 2*n - 2

# Standard error, t-Statistic and  p-value for Slope "alpha" coefficient
SE_alpha = np.sqrt(RSE**2/rl['VarX'].sum())
t_alpha = alpha/SE_alpha
p_alpha = 1 - stats.t.cdf(t_alpha,df=df)

# Standard error, t-Statistic and  p-value for Intercept "beta" coefficient
SE_beta = np.sqrt(RSE*(1/n + xmean**2/(rl['VarX'].sum())))
t_beta = beta/SE_beta 
p_beta = 1 - stats.t.cdf(t_beta,df=df)


# ## Coefficients Assessment Summary


# Assessment of Coefficients
mds = pd.DataFrame(
    {'Name':['Slope (alpha)', 'Intercept (beta)'],
     'Coefficient': [alpha, beta],
     'RSE':[SE_alpha, SE_beta],
     't-Statistic':[t_alpha, t_beta],
     'p-Value':[p_alpha, p_beta]
    }
)
mds


# ## Model Assessment Summary

# Model Assessment - Storing all key indicators in dummy data frame with range 1
ms = pd.DataFrame(
    {'Ref': range(0,1),
     'Residual Sum of Squares (RSS)': RSS,
     'RSE (Standard Deviation σ)': RSE,
     'Total Sum of Squares (TSS)': TSS,
     'R2 Statistic': R2
     }
)

# Cut out the dummy index column to see the Results
ms.iloc[:,1:9]    


# ## Plot Predicted vs Actual vs Sampled Data

# Plot regression against actual data
plt.figure(figsize=(12, 6))
# Population Regression Line
plt.plot(X,rl['RegL'], label = 'Actual (Population Regression Line)',color='green')
# Least squares line
plt.plot(X, ypred, label = 'Predicted (Least Squares Line)', color='blue')     
# scatter plot showing actual data
plt.plot(X, y, 'ro', label ='Collected data')   
plt.title('Actual vs Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()


# # B. Solve using Matrix Algebra - Fixed data points

# ## Create and Format Data


# Reuse the same random inputs created above but reformated to Matrices
X1 = np.matrix([np.ones(n), rl['X']]).T
y1 = np.matrix(rl['y']).T


# ## Solve for projection matrix


A = np.linalg.inv(X1.T.dot(X1)).dot(X1.T).dot(y1)

m = np.asscalar(A[1])
b = np.asscalar(A[0])

print("b (bias/Y intercept) =",b,", and m (slope) =",m)


# ## Plot data and predictions


#xx = np.linspace(0, .5, 2)
y1pred = b + m * X


# Plot regression against actual data
plt.figure(figsize=(12, 6))
# Population Regression Line
plt.plot(X,rl['RegL'], label = 'Actual (Population Regression Line)',color='green')
# Least squares line
plt.plot(X, y1pred, label = 'Predicted (Least Squares Line)', color='blue')     
# scatter plot showing actual data
plt.plot(X, y, 'ro', label ='Collected data')   
plt.title('Actual vs Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()






