#4) Salary_hike -> Build a prediction model for Salary_hike
#Salary_Data.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load data set
sal=pd.read_csv(r"provide path\Salary_Data.csv")

#data preprocessing
#view data set
sal

#changing column names for ease of operation
sal.columns="Exp","Salary"


#Exploratory data analysis
plt.hist(sal.Salary)
plt.boxplot(sal.Salary,0,"rs",0)


plt.hist(sal.Exp)
plt.boxplot(sal.Exp)

#sctterplot between Salaryhike and Experience
plt.plot(sal.Exp,sal.Salary,"bo");plt.xlabel("Experience");plt.ylabel("Salary")
#high correlation

sal.Salary.corr(sal.Exp) # # correlation value between X and Y
#very high 0.978

# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model1=smf.ols("Salary~Exp",data=sal).fit()
#ols- ordinary least square
# For getting coefficients of the varibles used in equation
model1.params


# P-values for the variables and R-squared value for prepared model
model1.summary()
#Rsq value=0.957
#AIC aikkaki information criteria should be minimum for good model
#std error =sigma/sqrt(n) sigma-sample 


print(model1.conf_int(0.05)) # 95% confidence interval


pred = model1.predict(sal.iloc[:,0]) # Predicted values of SalaryHike using the model
pred
# Visualization of regresion line over the scatter plot of Waist and AT
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=sal['Exp'],y=sal['Salary'],color='yellow');plt.plot(sal['Exp'],pred,color='black');plt.xlabel('Experience');plt.ylabel("Salaryhike")
#high correlation

pred.corr(sal.Salary) # very high 0.978

import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(model1.resid_pearson, dist="norm", plot=pylab)
#data are normal
#checking for residuals
plt.hist(model1.resid_pearson)
plt.boxplot(model1.resid_pearson)
np.median(model1.resid_pearson)
#value is -0.079 which is almost zero

# we select model1 as final model as all parametrs for model are very promising
#no need for transformations