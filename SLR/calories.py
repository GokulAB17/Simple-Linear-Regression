#1) Calories_consumed-> predict weight gained using calories consumed
#calories_consumed.csv datset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load data set
cal=pd.read_csv(r"provide_path\calories_consumed.csv")

#data preprocessing
#view data set
cal
#changing column names for ease of operation
cal.columns="weight","calories"

#Exploratory data analysis
plt.hist(cal.weight) # plotting histogram for weight col
plt.boxplot(cal.weight,0,'rs',0)#plotting box plot 

plt.hist(cal.calories)#histogram and boxplot for calories
plt.boxplot(cal.calories)

#Scatter plot between calories to weight
plt.plot(cal.calories,cal.weight,"bo");plt.xlabel("caloriesconsumed");plt.ylabel("weightgain")


cal.calories.corr(cal.weight) # # correlation value between X and Y
#corr=0.947  high collinearity

# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model1=smf.ols("weight~calories",data=cal).fit()
#ols- ordinary least square

# For getting coefficients of the varibles used in model
model1.params

# P-values for the variables and R-squared value for prepared model
model1.summary()
#Rsq value =0.897 high value
#AIC aikkaki information criteria should be minimum for good model
#std error =sigma/sqrt(n) sigma-sample 
help("numpy.std")

print(model1.conf_int(0.05)) # 95% confidence interval


pred = model1.predict(cal.iloc[:,1]) # Predicted values of weight using the model
pred
# Visualization of regresion line over the scatter plot of calories and weight
plt.scatter(x=cal['calories'],y=cal['weight'],color='red');plt.plot(cal['calories'],pred,color='black');plt.xlabel('calories');plt.ylabel('weight')
#good fitting of all data points over regression


#correlation value between predicted values and actual values of weight
pred.corr(cal.weight) # 0.81
#good correlation

#checking residuals 
# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(model1.resid_pearson, dist="norm", plot=pylab)
plt.hist(model1.resid_pearson)
plt.boxplot(model1.resid_pearson)
np.mean(model1.resid_pearson)
#mean is 1.078e-15 <<<0 

#doing transformation for better R values 
#logarithmic transformations 
model2=smf.ols("weight~np.log(calories)",data=cal).fit()
model2.params
model2.summary()
#Rsq value 0.808 which is less than previous model1
print(model2.conf_int(0.05)) # 95% confidence level
pred2 = model2.predict(pd.DataFrame(cal['calories']))
pred2.corr(cal.weight)
#0.89 good correlation

#checking for residuals
# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(model2.resid_pearson, dist="norm", plot=pylab)
plt.hist(model2.resid_pearson)# data  normal along mean
plt.boxplot(model2.resid_pearson)
np.mean(model2.resid_pearson)
#mean is -3.18e-15 <<< 0

#exponential transformation
model3=smf.ols("np.log(weight)~calories",data=cal).fit()
model3.params
model3.summary()
#Rsq value 0.878 which is less than model1

#we select model1 for prediction for weightgain within the 
#calories consumed range of given set of values




