
import pandas as pd
import sklearn
import numpy
import scipy
import statsmodels
import math
import matplotlib as matlab
import matplotlib.pyplot as plt
import PIL

#######################################################
########LAB: Simple Classifiers

#Dataset: Fraud Transaction/Transactions_sample.csv
Transactions_sample = pd.read_csv("C:\\Koti\\data science\\data\\drive-download-20160927T020851Z\\Fraud Transaction\\Transactions_sample.csv")

Transactions_sample.head(6)
Transactions_sample.columns

#Draw a classification graph that shows all the classes
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(Transactions_sample.Total_Amount[Transactions_sample.Fraud_id==0],Transactions_sample.Tr_Count_week[Transactions_sample.Fraud_id==0], s=30, c='b', marker="o", label='Fraud_id 0')
ax1.scatter(Transactions_sample.Total_Amount[Transactions_sample.Fraud_id==1],Transactions_sample.Tr_Count_week[Transactions_sample.Fraud_id==1], s=30, c='r', marker="+", label='Fraud_id 1')

plt.xlim(min(Transactions_sample.Total_Amount), max(Transactions_sample.Total_Amount))
plt.ylim(min(Transactions_sample.Tr_Count_week), max(Transactions_sample.Tr_Count_week))

plt.legend(loc='upper left');

plt.show()



#Build a logistic regression classifier 

import statsmodels.formula.api as sm
model1 = sm.logit(formula='Fraud_id ~ Total_Amount+Tr_Count_week', data=Transactions_sample)
fitted1 = model1.fit()
fitted1.params

fitted1.params[0]
fitted1.params[1]
fitted1.params[2]


#Draw the classifier on the data plot

# getting slope and intercept of the line
slope1=fitted1.params[1]/(-fitted1.params[2])
intercept1=fitted1.params[0]/(-fitted1.params[2])


import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(Transactions_sample.Total_Amount[Transactions_sample.Fraud_id==0],Transactions_sample.Tr_Count_week[Transactions_sample.Fraud_id==0], s=30, c='b', marker="o", label='Fraud_id 0')
ax1.scatter(Transactions_sample.Total_Amount[Transactions_sample.Fraud_id==1],Transactions_sample.Tr_Count_week[Transactions_sample.Fraud_id==1], s=30, c='r', marker="+", label='Fraud_id 1')

plt.xlim(min(Transactions_sample.Total_Amount), max(Transactions_sample.Total_Amount))
plt.ylim(min(Transactions_sample.Tr_Count_week), max(Transactions_sample.Tr_Count_week))

plt.legend(loc='upper left');

x_min, x_max = ax1.get_xlim()
ax1.plot([0, x_max], [intercept1, x_max*slope1+intercept1])
plt.show()

#Accuracy of the model
#Creating the confusion matrix
predicted_values=fitted1.predict(Transactions_sample[["Total_Amount"]+["Tr_Count_week"]])
predicted_values[1:10]
threshold=0.5

import numpy as np
predicted_class=np.zeros(predicted_values.shape)
predicted_class[predicted_values>threshold]=1

predicted_class

from sklearn.metrics import confusion_matrix as cm
ConfusionMatrix = cm(Transactions_sample[['Fraud_id']],predicted_class)
print(ConfusionMatrix)

accuracy=(ConfusionMatrix[0,0]+ConfusionMatrix[1,1])/sum(sum(ConfusionMatrix))
print(accuracy)

error=1-accuracy
print(error)


#Build a SVM classifier 

######################################
######SVM Models
#####################################
#Dataset: Fraud Transaction/Transactions_sample.csv

#Build a SVM classifier 

X = Transactions_sample[['Total_Amount']+['Tr_Count_week']]  
y = Transactions_sample['Fraud_id']
from sklearn.svm import SVC
clf = SVC()
model =clf.fit(X,y)
clf.score(X,y)
Predicted = numpy.zeros(50)

# NOTE: If i is in range(0,n), then i takes vales [0,n-1] 
for i in range(0,50):
    a = Transactions_sample.Total_Amount[i]
    b = Transactions_sample.Tr_Count_week[i]
    Predicted[i]=clf.predict([[a,b]])
    del a,b

#Draw the classifier on the data plots
#Plotting in SVM

#Predict the (Fraud vs not-Fraud) class for the data points Total_Amount=11000, Tr_Count_week=15 & Total_Amount=2000, Tr_Count_week=4
#Prediction in SVM
new_data1=[11000, 15]
new_data2=[2000,4]

#Predict the (Fraud vs not-Fraud) class for the data points Total_Amount=11000, Tr_Count_week=15 & Total_Amount=2000, Tr_Count_week=4
NewPredicted1=model.predict([new_data1])
print(NewPredicted1)

NewPredicted2=clf.predict([new_data2])
print(NewPredicted2)   

#Download the complete Dataset: Fraud Transaction/Transaction.csv
#SVM on overall data
Transactions= pd.read_csv("C:\\Koti\\data science\\data\\drive-download-20160927T020851Z\\Fraud Transaction\\Transaction.csv")

#Draw a classification graph that shows all the classes



#Converting the output into factor, otherwise SVM will fit a regression model
#Build a SVM classifier 
clf = SVC()
X = Transactions[['Total_Amount']+['Tr_Count_week']]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = Transactions[['Fraud_id']]

Smodel =clf.fit(X,y)
    
###Draw the classifier on the data plots
#Plotting SVM Clasification graph
    


#### LAB: Kernel Non-Linear Classifier
######################################
###### Kernels
#Dataset : Software users/sw_user_profile.csv  
import pandas as pd
sw_user_profile = pd.read_csv("C:\\Koti\\data science\\data\\drive-download-20160927T020851Z\\Software users\\sw_user_profile.csv")
#How many variables are there in software user profile data?
sw_user_profile.shape
sw_user_profile.columns
# 3 Variables
#Plot the active users against and check weather the relation between age and "Active" status is linear or non-linear
plt.scatter(sw_user_profile.Age,sw_user_profile.Id,color='blue')



#Build an SVM model(model-1), make sure that there is no kernel or the kernel is linear
from sklearn.svm import SVC
#Model Building 
X= sw_user_profile[['Age']]
y= sw_user_profile[['Active']]
clf = SVC()
Smodel =clf.fit(X,y)


#Making the kernel to linear

Linsvc = SVC(kernel='linear', C=1).fit(X, y)
predict3 = Linsvc.predict(X)
#For model-1, create the confusion matrix and find out the accuracy
#Confusion Matrix
from sklearn.metrics import confusion_matrix 
conf_mat = confusion_matrix(sw_user_profile[['Active']],predict3)

help(SVC.predict)

#######################################
#New variable derivation. Mapping to higher dimensions

#Standardizing the data to visualize the results clearly
sw_user_profile['age_nor']=(sw_user_profile.Age-numpy.mean(sw_user_profile.Age))/numpy.std(sw_user_profile.Age) 


#Create a new variable. By using the polynomial kernel
#Creating the new variable
sw_user_profile['new']=(sw_user_profile.age_nor)*(sw_user_profile.age_nor)
   




#Build an SVM model(model-2), with the new data mapped on to higher dimensions. Keep the default kernel in R as linear

#Model Building with new variable
X= sw_user_profile[['Age']+['new']]
y= sw_user_profile[['Active']]
Linsvc = SVC(kernel='linear', C=1).fit(X, y)
predict4 = Linsvc.predict(X)
#For model-2, create the confusion matrix and find out the accuracy
#Confusion Matrix
conf_mat = confusion_matrix(sw_user_profile[['Active']],predict4)







#With the original data re-cerate the model(model-3) and let python choose the default kernel function. 
########Model Building with radial kernel function
X= sw_user_profile[['Age']]
y= sw_user_profile[['Active']]
Linsvc = SVC(kernel='rbf', C=1).fit(X, y)
predict5 = Linsvc.predict(X)
conf_mat = confusion_matrix(sw_user_profile[['Active']],predict5)

# performing the SVM on Fiberbits data
Fiberbits=Fiberbits[1:20000]
Fiberbits.shape # checking dimensions
Fiberbits.columns
X=Fiberbits[['income','months_on_network','Num_complaints','number_plan_changes', 'relocated', 'monthly_bill',
       'technical_issues_per_month', 'Speed_test_result']]
y= Fiberbits[['active_cust']]

Linsvc = SVC(kernel='rbf', C=1).fit(X, y)
predict5 = Linsvc.predict(X)
conf_mat = confusion_matrix(sw_user_profile[['Active']],predict5)
