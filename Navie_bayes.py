# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 10:29:20 2018

@author: Koti
"""

from sklearn.naive_bayes import GaussianNB
import numpy as np
model = GaussianNB()
features=fb.drop(["active_cust"],axis=1)
X_train=features
y_train=fb["active_cust"]
model.fit(X_train,y_train)
model.score(X_train,y_train)
model.predict(X_train)
model.class_count_
fb["active_cust"].value_counts()
model.class_prior_
model.classes_
model.partial_fit
model.priors
model.theta_
