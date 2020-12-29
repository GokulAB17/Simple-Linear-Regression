#3) Emp_data -> Build a prediction model for Churn_out_rate 
#emp_data.csv dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load data set
chrate=pd.read_csv(r"provide path\emp_data.csv")

#data preprocessing
#view data set
chrate
#changing column names for ease of operation
chrate.columns="Salary","Churnrate"

#Exploratory data analysis
plt.hist(chrate.Salary)
plt.boxplot(chrate.Salary,0,"rs",0)


plt.hist(chrate.Churnrate)
plt.boxplot(chrate.Churnrate)

#Scatter plot for Churnrate and Salary
plt.plot(chrate.Salary,chrate.Churnrate,"bo");plt.xlabel("SalaryHike");plt.ylabel("Churnoutrate")


chrate.Salary.corr(chrate.Churnrate) # # correlation value between X and Y
#cor is high 0.911

# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model1=smf.ols("Churnrate~Salary",data=chrate).fit()
#ols- ordinary least square
# For getting coefficients of the varibles used in equation
model1.params
type(model1)

# P-values for the variables and R-squared value for prepared model
model1.summary()
#Rsq value = 0.831
#AIC aikkaki information criteria should be minimum for good model
#std error =sigma/sqrt(n) sigma-sample 


print(model1.conf_int(0.05)) # 95% confidence interval


pred = model1.predict(chrate.iloc[:,0]) # Predicted values of Churnrate using the model
pred

# Visualization of regresion line over the scatter plot of Churnrate and Salary
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=chrate['Salary'],y=chrate['Churnrate'],color='yellow');plt.plot(chrate['Salary'],pred,color='black');plt.xlabel('Salaryhike');plt.ylabel('Churnoutrate')

pred.corr(chrate.Churnrate) 
# cor value high 0.911

#Checking for residuals
#checking residuals for model1 
#checking residuals 
# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(model1.resid_pearson, dist="norm", plot=pylab)
plt.hist(model1.resid_pearson)
plt.boxplot(model1.resid_pearson)
np.mean(model1.resid_pearson)
#mean -6.44e-16=== 0


#Logarithmic transformations
model2=smf.ols("Churnrate~np.log(Salary)",data=chrate).fit()

model2.params
model2.summary()
#Rsq value is 0.849 improved from 0.831

#Exponential Transformations
model3=smf.ols("Churnrate~np.exp(Salary)",data=chrate).fit()
model3.params
model3.summary()
#Rsq value is 0.361 

model4=smf.ols("Churnrate~Salary*Salary",data=chrate).fit()
model4.params
model4.summary()
#Rsq value is 0.831

# we select logarithmic model model2 for prediction purpose for churnrate
pred2 = model2.predict(chrate.iloc[:,0]) # Predicted values of Churnrate using the model
pred

# Visualization of regresion line over the scatter plot of Churnrate and Salary
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=chrate['Salary'],y=chrate['Churnrate'],color='yellow');plt.plot(chrate['Salary'],pred2,color='black');plt.xlabel('Salaryhike');plt.ylabel('Churnoutrate')

pred.corr(chrate.Churnrate) 
# cor value high 0.911

#Checking for residuals
#checking residuals for model2

# Checking Residuals are normally distributed
st.probplot(model1.resid_pearson, dist="norm", plot=pylab)
plt.hist(model1.resid_pearson)
plt.boxplot(model1.resid_pearson)
np.mean(model1.resid_pearson)
