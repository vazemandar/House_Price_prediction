# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:38:41 2018

@author: mandar
"""

# import the data Set and Take a look at the data
import pandas as pd
import numpy as np
from sklearn.linear_model import  Lasso
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
data=pd.read_csv('C:/Users/mandar/Downloads/train_corrected.csv')

# these variables have just one value for all the records
# These features are not going to be use for us in the study
Drop_columns=[]
for name in data.columns:
    if (len(data[name].unique())<2):
        Drop_columns.append(name)

#Lets perform the first model for year 2003
data_2003=data.loc[data['AHSYEAR'] == 2003]

data_2003['YearsOld']=data_2003['AHSYEAR']-data_2003['BUILT']    

data_2003=data_2003[['CONTROL','VALUE','YearsOld','BATHS','ROOMS','CELLAR','BEDRMS','UNITSF','FAMRM','DINING','HALFB','FLOORS','FPLWK','DISH','DISPL','OWNLOT','INCP','NUNIT2','GARAGE']]
data_2005=data.loc[data['AHSYEAR'] == 2005]
data_2005['YearsOld']=data_2005['AHSYEAR']-data_2005['BUILT']    
data_2005=data_2005[['CONTROL','VALUE','YearsOld','BATHS','ROOMS','CELLAR','BEDRMS','UNITSF','FAMRM','DINING','HALFB','FLOORS','FPLWK','DISH','DISPL','OWNLOT','INCP','NUNIT2','GARAGE']]
data_2007=data.loc[data['AHSYEAR'] == 2007]        
data_2007['YearsOld']=data_2007['AHSYEAR']-data_2007['BUILT']    
data_2007=data_2007[['CONTROL','VALUE','YearsOld','BATHS','ROOMS','CELLAR','BEDRMS','UNITSF','FAMRM','DINING','HALFB','FLOORS','FPLWK','DISH','DISPL','OWNLOT','INCP','NUNIT2','GARAGE']]

 
Train_y_2003 = data_2003[['VALUE','CONTROL']]


Train_y_2005 = data_2005[['VALUE','CONTROL']]
Train_y_2007 = data_2007[['VALUE','CONTROL']]

Train_y= pd.merge(Train_y_2003, Train_y_2005,on='CONTROL')

Train_y= pd.merge(Train_y, Train_y_2007,on='CONTROL')
Train_y = Train_y.drop(['CONTROL'], axis=1)



data_2003 = data_2003.drop(['VALUE'], axis=1)
data_2005 = data_2005.drop(['VALUE'], axis=1)
data_2007 = data_2007.drop(['VALUE'], axis=1)




Train_X= pd.merge(data_2003, data_2005,on='CONTROL')

Train_X= pd.merge(Train_X, data_2007,on='CONTROL')

Train_X = Train_X.drop(['CONTROL'], axis=1)

#Train_X = Train_X[['YearsOld','BATHS','ROOMS','CELLAR','BEDRMS','UNITSF','FAMRM','DINING','HALFB','FLOORS','FPLWK','DISH','DISPL','OWNLOT','INCP','NUNIT2','GARAGE']]
#To check the RMSE values for differnet models 
train_data, test_data,train_label,test_label  = train_test_split(Train_X,Train_y, test_size=0.1,random_state=0)
n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(Train_X)
    rmse= np.sqrt(-cross_val_score(model, train_data, train_label, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

  #Variefying the first Model

GBoost = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=3000, learning_rate=0.07,
                                   max_depth=5, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5))

GBoost.fit(train_data, train_label)
xgb_train_pred = GBoost.predict(test_data)
#score = rmsle_cv(GBoost)
#print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))    
