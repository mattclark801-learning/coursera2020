# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:53:44 2020

@author: mattc

This program will demonstrate an ANOVA test and Post Hoc test (Multiple Comparison of Means - Tukey HSD).
To perform the ANOVA test we require a categorical explanatory variable and a quantatative response variable (C -> Q). We discard the variables used for the Course 1 project and choose new variables.

C: W1_A5A: Who did you vote for?: Categorical Variable
Q: W2_DURATION: Quantitative variable


"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 

df = pd.read_csv('mycodebook.csv', low_memory=False)

# using ols function for calculating the F-statistic and associated p value

model1 = smf.ols(formula='W2_DURATION ~ C(W1_A5A)', data=df)
results1 = model1.fit()
print (results1.summary())

df1 = df[['W2_DURATION', 'W1_A5A']].dropna()

#calculate means

print ('means for W2_DURATION by W1_A5A status')
m1= df1.groupby('W1_A5A').mean()
print (m1)

#calculate standard deviations

print ('standard deviations for W2_DURATION by W1_A5A status')
sd1 = df1.groupby('W1_A5A').std()
print (sd1)

#tukey HSD post-hoc test

mc1 = multi.MultiComparison(df1['W2_DURATION'], df1['W1_A5A'])
res1 = mc1.tukeyhsd()
print(res1.summary())

