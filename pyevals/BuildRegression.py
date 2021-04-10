import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression ,Ridge , Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import ElasticNet 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor  
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score , mean_squared_error,mean_absolute_error
from pyevals import AdjustedR2 as ar
import math

import warnings 
warnings.filterwarnings('ignore')

def metrics(x_train,y_test,predicted):

	R2_score = 1 - r2_score(y_test,predicted)
	AdjustedR2 = 1 - ar.AdjustedR2(R2_score,x_train)
	MeanAbsoluteError = mean_absolute_error(y_test,predicted)
	MeanSquaredError = mean_squared_error(y_test,predicted)
	RootMeanSquaredError = math.sqrt(MeanSquaredError)

	return R2_score,AdjustedR2,MeanAbsoluteError,MeanSquaredError,RootMeanSquaredError


def LinearReg(x_train,x_test,y_train,y_test):

	model = LinearRegression()
	model.fit(x_train, y_train)
	predicted = model.predict(x_test)

	return metrics(x_train,y_test,predicted)

def PolynomialRegression(x_train,x_test,y_train,y_test):

	Input=[('polynomial',PolynomialFeatures(degree=2)),('modal',LinearRegression())]
	model=Pipeline(Input)
	model.fit(x_train,y_train)
	predicted = model.predict(x_test)

	return metrics(x_train,y_test,predicted)

def RidgeRegression(x_train,x_test,y_train,y_test):

	model = Ridge()
	model.fit(x_train,y_train)
	predicted = model.predict(x_test)

	return metrics(x_train,y_test,predicted)

def LassoRegression(x_train,x_test,y_train,y_test):

	model = Lasso()
	model.fit(x_train,y_train)
	predicted = model.predict(x_test)

	return metrics(x_train,y_test,predicted)

def SupportVectorRegressor(x_train,x_test,y_train,y_test):

	model = SVR()
	model.fit(x_train,y_train)
	predicted = model.predict(x_test)

	return metrics(x_train,y_test,predicted)

def GradientBoostingRegression(x_train,x_test,y_train,y_test):

	model = GradientBoostingRegressor()
	model.fit(x_train,y_train)
	predicted = model.predict(x_test)

	return metrics(x_train,y_test,predicted)

def PartialLeastSquares(x_train,x_test,y_train,y_test):
	model = PLSRegression()
	model.fit(x_train,y_train)
	predicted = model.predict(x_test)
	return metrics(x_train,y_test,predicted)
