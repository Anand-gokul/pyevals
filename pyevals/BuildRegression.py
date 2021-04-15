from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pyevals.RegressionMetrics import *
import math

import warnings

warnings.filterwarnings('ignore')


def metrics(X_train, y_test, predicted):
	R2_score = 1 - rsquared_score(y_test, predicted)
	AdjustedR2 = 1 - adj_rsquared_score(X_train, y_test, predicted, R2_score)
	MeanAbsoluteError = mean_absolute_error(y_test, predicted)
	MeanSquaredError = mean_squared_error(y_test, predicted)
	RootMeanSquaredError = math.sqrt(MeanSquaredError)
	MeanAbsolutePercentageError = mean_absolute_percentage_error(y_test, predicted)

	return R2_score, AdjustedR2, MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError, MeanAbsolutePercentageError


def LinearReg(X_train, X_test, y_train, y_test=None):
	if y_test is not None:
		model = LinearRegression()
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return metrics(X_train, y_test, predicted)
	else:
		model = LinearRegression()
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return predicted


def PolynomialRegression(X_train, X_test, y_train, y_test=None):
	if y_test is not None:
		Input = [('polynomial', PolynomialFeatures(degree=2)), ('modal', LinearRegression())]
		model = Pipeline(Input)
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return metrics(X_train, y_test, predicted)
	else:
		Input = [('polynomial', PolynomialFeatures(degree=2)), ('modal', LinearRegression())]
		model = Pipeline(Input)
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return predicted


def RidgeRegression(X_train, X_test, y_train, y_test=None):
	if y_test is not None:
		model = Ridge()
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return metrics(X_train, y_test, predicted)
	else:
		model = Ridge()
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return predicted


def LassoRegression(X_train, X_test, y_train, y_test=None):
	if y_test is not None:
		model = Lasso()
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return metrics(X_train, y_test, predicted)
	else:
		model = Lasso()
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return predicted


def SupportVectorRegressor(X_train, X_test, y_train, y_test=None):
	if y_test is not None:
		model = SVR()
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return metrics(X_train, y_test, predicted)
	else:
		model = SVR()
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return predicted


def GradientBoostingRegression(X_train, X_test, y_train, y_test=None):
	if y_test is not None:
		model = GradientBoostingRegressor()
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return metrics(X_train, y_test, predicted)
	else:
		model = GradientBoostingRegressor()
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return predicted


def PartialLeastSquares(X_train, X_test, y_train, y_test=None):
	if y_test is not None:
		model = PLSRegression()
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return metrics(X_train, y_test, predicted)
	else:
		model = PLSRegression()
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return predicted


def KNNRegressor(X_train, X_test, y_train, y_test=None):
	if y_test is not None:
		model = KNeighborsRegressor()
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return metrics(X_train, y_test, predicted)
	else:
		model = KNeighborsRegressor()
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return predicted


def RFRegressor(X_train, X_test, y_train, y_test=None):
	if y_test is not None:
		model = RandomForestRegressor()
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return metrics(X_train, y_test, predicted)
	else:
		model = RandomForestRegressor()
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return predicted


def DTRegressor(X_train, X_test, y_train, y_test):
	if y_test is not None:
		model = DecisionTreeRegressor()
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return metrics(X_train, y_test, predicted)
	else:
		model = DecisionTreeRegressor()
		model.fit(X_train, y_train)
		predicted = model.predict(X_test)

		return predicted