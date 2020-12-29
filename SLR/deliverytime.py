#2) Delivery_time -> Predict delivery time using sorting time 
#delivery_time.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#load data set
del_time=pd.read_csv(r"provide_path\delivery_time.csv")

#data preprocessing
#view data set
del_time

#changing column names for ease of operation
del_time.columns="Deltime","Sorttime"

#Exploratory data analysis
plt.hist(del_time.Deltime)#histogram and boxplot for Delivery time
plt.boxplot(del_time.Deltime,0,'rs',0)


plt.hist(del_time.Sorttime)#histogram and boxplot for Sorting time
plt.boxplot(del_time.Sorttime)

#Scatter plot between sorting time to delivery time
plt.plot(del_time.Sorttime,del_time.Deltime,"bo");plt.xlabel("SortingTime");plt.ylabel("DeliveryTime")
#moderate corelation from plot

del_time.Deltime.corr(del_time.Sorttime) # # correlation value between X and Y
#corr=0.825

# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model1=smf.ols("Deltime~Sorttime",data=del_time).fit()


# For getting coefficients of the varibles used in equation
model1.params

# P-values for the variables and R-squared value for prepared model
model1.summary()
#Rsq value= 0.682 lesser value so we ll require transformations
#AIC aikkaki information criteria should be minimum for good model
#std error =sigma/sqrt(n) sigma-sample 

print(model1.conf_int(0.05)) # 95% confidence interval


pred = model1.predict(del_time.iloc[:,1]) # Predicted values of delivery time using the model
pred
# Visualization of regresion line over the scatter plot of Sort time and delivery time
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=del_time['Sorttime'],y=del_time['Deltime'],color='yellow');plt.plot(del_time['Sorttime'],pred,color='black');plt.xlabel('Sortingtime');plt.ylabel('Deliverytime')

pred.corr(del_time.Deltime) # 0.826
#strong correlation

#logarithmic transformation
model2=smf.ols("Deltime~np.log(Sorttime)",data=del_time).fit()
model2.params
model2.summary()
#Rsq value 0.695 better than model1
#p vlaue for intercept 0.642 high than 0.05
print(model2.conf_int(0.05)) # 99% confidence level
pred2 = model2.predict(pd.DataFrame(del_time['Sorttime']))
pred2.corr(del_time.Deltime)
#corr 0.833

#exponential transformation
model3=smf.ols("Deltime~np.exp(Sorttime)",data=del_time).fit()
model3.params
model3.summary()
#Rsq value very less 0.361
#go for another transformation

#quadratic transformations
model4=smf.ols("Deltime~Sorttime*Sorttime",data=del_time).fit()
model4.params
model4.summary()
#Rsq is 0.682
print(model4.conf_int(0.05)) # 99% confidence level
pred4 = model4.predict(pd.DataFrame(del_time['Sorttime']))
pred4.corr(del_time.Deltime)
#strong corr 0.826

#model2 has comparatively higher value than model1 & model2 
#but p value of intercept is very high so we should go for model 1 
#for bettering R sq value increase no.of records
 
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
#mean is -4.229e-16 === zero

