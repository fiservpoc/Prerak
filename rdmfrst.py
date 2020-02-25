# -*- coding: utf-8 -*-

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def rf( df,features_train,labels_train,features_test,labels_test ):
    
    print('INSIDE RF File')
    print(features_train.shape)
    
    ################## Check for Default Parameters    
    #rf_0 = RandomForestClassifier(random_state = 8)
    #print('Parameters currently in use:\n')
    #pprint(rf_0.get_params())
    
    ################## Extract Best Parameters from Random Search
    ## set parameters
   
    # n_estimators
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]
    
    # max_features
    max_features = ['auto', 'sqrt']
    
    # max_depth
    max_depth = [int(x) for x in np.linspace(20, 100, num = 5)]
    max_depth.append(None)
    
    # min_samples_split
    min_samples_split = [2, 5, 10]
    
    # min_samples_leaf
    min_samples_leaf = [1, 2, 4]
    
    # bootstrap
    bootstrap = [True, False]
    
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    print('Random Parameters \n')
    pprint(random_grid)
    
    # First create the base model to tune
    rfc = RandomForestClassifier(random_state=8)
    
    # Definition of the random search
    random_search = RandomizedSearchCV(estimator=rfc,
                                       param_distributions=random_grid,
                                       n_iter=50,
                                       scoring='accuracy',
                                       cv=3, 
                                       verbose=1, 
                                       random_state=8)
    
    # Fit the random search model
    random_search.fit(features_train, labels_train)
    print("The best hyperparameters from Random Search are:")
    pprint(random_search.best_params_)
    print("")
    print("The mean accuracy of a model with these hyperparameters is:")
    print(random_search.best_score_)
        