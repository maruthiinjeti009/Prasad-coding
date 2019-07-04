##############################################################################
####################### logistic Regresssion ###################################

##### LAB-What is the need of logistic regression ################

#Import the Dataset: Product Sales Data/Product_sales.csv
import numpy as np
import pandas as pd
import matplotlib as plt
import scipy as sp

sales=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\Product_sales.csv")

#What are the variables in the dataset? 
sales.columns.values

#Build a predictive model for Bought vs Age

### we need to use the statsmodels package, which enables many statistical methods to be used in Python
import statsmodels.formula.api as sm
from statsmodels.formula.api import ols
model = sm.ols(formula='Bought ~ Age', data=sales)
fitted = model.fit()
fitted.summary()

#If Age is 4 then will that customer buy the product?

import sklearn as sk
from sklearn import linear_model

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(sales[["Age"]], sales[["Bought"]])

age1=4
predict1=lr.predict(age1)
predict1
age2=65
predict2=lr.predict(age2)
predict2

### for age=4 value is less than zero  

#If Age is 105 then will that customer buy the product?
age2=105
predict2=lr.predict(age2)
predict2
##### for age=105 value is greater than one.

#######From this linear regression,we can not interpret whether a person buys or not

################ Lab: Logistic Regression ######################

#Dataset: Product Sales Data/Product_sales.csv
sales=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\Product_sales.csv")

# Build a logistic Regression line between Age and buying 
import statsmodels.formula.api as sm
logit=sm.Logit(sales['Bought'],sales['Age'])
logit
result = logit.fit()
result
result.summary2()

###coefficients Interval of each coefficient
print (result.conf_int())

#One more way of fitting the model
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(sales[["Age"]],sales["Bought"])

#A 4 years old customer, will he buy the product?
age1=4
predict_age1=logistic.predict(age1)
print(predict_age1)

#If Age is 105 then will that customer buy the product?
age2=105
predict_age2=logistic.predict(age2)
print(predict_age2)

##############LAB: Multiple Logistic Regression####################

#Dataset: Fiberbits/Fiberbits.csv
Fiber=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\Fiberbits.csv")
list(Fiber.columns.values)  ###to get variables list

#Build a model to predict the chance of attrition for a given customer using all the features. 
from sklearn.linear_model import LogisticRegression
logistic1= LogisticRegression()
###fitting logistic regression for active customer on rest of the varibles#######
logistic1.fit(Fiber[["income"]+['months_on_network']+['Num_complaints']+['number_plan_changes']+['relocated']+['monthly_bill']+['technical_issues_per_month']+['Speed_test_result']],Fiber[['active_cust']])

predict1=logistic1.predict(Fiber[["income"]+['months_on_network']+['Num_complaints']+['number_plan_changes']+['relocated']+['monthly_bill']+['technical_issues_per_month']+['Speed_test_result']])
predict1

predict1

#How good is your model?
### calculate confusion matrix
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix###for using confusion matrix###
cm1 = confusion_matrix(Fiber[['active_cust']],predict1)
print(cm1)
sum(cm1)
total1=sum(sum(cm1))
total1

#####from confusion matrix calculate accuracy
accuracy1=(cm1[0,0]+cm1[1,1])/total1
accuracy1

#What are the most impacting variables?
#### From summary of the model

logit1=sm.Logit(Fiber['active_cust'],Fiber[['income']+['months_on_network']+['Num_complaints']+['number_plan_changes']+['relocated']+['monthly_bill']+['technical_issues_per_month']+['Speed_test_result']])
logit1
result1=logit1.fit()

result1.summary2()

#For all the variables p<0.05,so all are impacting

###############LAB: Confusion Matrix & Accuracy ########################

######same as above code ,find confusion matrix#####################
#Create confusion matrix for Fiber bits model

from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix###for using confusion matrix###
cm1 = confusion_matrix(Fiber[['active_cust']],predict1)
print(cm1)

#find the accuracy value for fiber bits model
total1=sum(sum(cm1))
#####from confusion matrix calculate accuracy
accuracy1=(cm1[0,0]+cm1[1,1])/total1
accuracy1

#Change try three different threshold values and note down the changes in accuracy value


###########  LAB-Multicollinearity################3
#Is there any multicollinearity in fiber bits model? 
#Identify and remove multicollinearity from the model

def vif_cal(input_data, dependent_col):
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.ols(formula="y~x", data=x_vars).fit().rsquared  
        vif=round(1/(1-rsq),2)
        print (xvar_names[i], " VIF = " , vif)

#Calculating VIF values using that function
vif_cal(input_data=Fiber, dependent_col="active_cust")


###################LAB: Individual Impact of Variables################

#Identify top impacting and least impacting variables in fiber bits models
####Variable importance is decided from Wald chi‐square value i.e square of Z parameter.
##from summary,sort the varibles in the order of the squares of their z values.
#Find the variable importance and order them based on their impact


########  LAB-Logistic Regression Model Selection ###########

#Find AIC and BIC values for the first fiber bits model(m1)
#Including all the variables
m1=sm.Logit(Fiber['active_cust'],Fiber[["income"]+['months_on_network']+['Num_complaints']+['number_plan_changes']+['relocated']+['monthly_bill']+['technical_issues_per_month']+['Speed_test_result']])
m1
m1.fit()
m1.fit().summary()
m1.fit().summary2()

#What are the top-2 impacting variables in fiber bits model?

#What are the least impacting variables in fiber bits model?
#Can we drop any of these variables and build a new model(m2)
#Income and Monthly Bill Dropped
m2=sm.Logit(Fiber['active_cust'],Fiber[['months_on_network']+['Num_complaints']+['number_plan_changes']+['relocated']+['technical_issues_per_month']+['Speed_test_result']])
m2
m2.fit()

m2.fit().summary2()

#We have two models, what the best accuracy that you can expect on this data

#Dropping high impacting variables 
#relocated and Speed_test_result dropped
m3=sm.Logit(Fiber['active_cust'],Fiber[["income"]+['months_on_network']+['Num_complaints']+['number_plan_changes']+['monthly_bill']+['technical_issues_per_month']])
m3
m3.fit()
m3.fit().summary()
m3.fit().summary2()

Fiber.info()

