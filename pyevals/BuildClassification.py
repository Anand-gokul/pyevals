import warnings
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.calibration import CalibratedClassifierCV

import warnings 
warnings.filterwarnings('ignore')


def metrics(y_test,Prediction):
    AccuracyScore = accuracy_score(y_test,Prediction)
    RecallScore = recall_score(y_test,Prediction,average='macro')
    PrecisionScore = precision_score(y_test,Prediction,average='macro')
    f1Score = f1_score(y_test,Prediction,average='macro')

    return AccuracyScore,RecallScore,PrecisionScore,f1Score

class my_warning(UserWarning):
    pass

def KNearestNeighbor(x_train,x_test,y_train,y_test):
    kNN = KNeighborsClassifier()
    kNN = kNN.fit(x_train,y_train)
    Prediction = kNN.predict(x_test)
    return metrics(y_test,Prediction)

def LogisticReg(x_train,x_test,y_train,y_test):
    Logistic_Regression = LogisticRegression()
    Logistic_Regression = Logistic_Regression.fit(x_train,y_train)
    Prediction = Logistic_Regression.predict(x_test)
    return metrics(y_test,Prediction)


def DecisionTree(x_train,x_test,y_train,y_test):
    DecisionTree_Classifier = DecisionTreeClassifier()
    DecisionTree_Classifier = DecisionTree_Classifier.fit(x_train,y_train)
    Prediction = DecisionTree_Classifier.predict(x_test)
    return metrics(y_test,Prediction)

def RandomForest(x_train,x_test,y_train,y_test):
    RandomForest_Classifier = RandomForestClassifier()
    RandomForest_Classifier = RandomForest_Classifier.fit(x_train,y_train)
    Prediction = RandomForest_Classifier.predict(x_test)
    return metrics(y_test,Prediction)

def SupportVector(x_train,x_test,y_train,y_test):
    SupportVectortClassifier = make_pipeline(SVC())
    SupportVectortClassifier.fit(x_train,y_train)
    Prediction = SupportVectortClassifier.predict(x_test)
    return metrics(y_test,Prediction)

def QuadraticDiscriminant(x_train,x_test,y_train,y_test):
    QuadraticDiscriminant_Analysis = QuadraticDiscriminantAnalysis()
    QuadraticDiscriminant_Analysis.fit(x_train,y_train)
    Prediction = QuadraticDiscriminant_Analysis.predict(x_test)
    return metrics(y_test,Prediction)
    
def SophisticatedGradientDescent(x_train,x_test,y_train,y_test):
    SGDClassifier_ = SGDClassifier()
    SGDClassifier_.fit(x_train,y_train)
    Prediction = SGDClassifier_.predict(x_test)
    return metrics(y_test,Prediction)

def AdaBoost(x_train,x_test,y_train,y_test):
    AdaBoostClassifier_ = AdaBoostClassifier()
    AdaBoostClassifier_.fit(x_train,y_train)
    Prediction = AdaBoostClassifier_.predict(x_test)
    return metrics(y_test,Prediction)

def CalibratedClassifier(x_train,x_test,y_train,y_test):
    CalibratedClassifierCV_ = CalibratedClassifierCV()
    CalibratedClassifierCV_ = CalibratedClassifierCV_.fit(x_train,y_train)
    Prediction = CalibratedClassifierCV_.predict(x_test) 
    return metrics(y_test,Prediction)

def GaussianNaiveBayes(x_train,x_test,y_train,y_test):
    Gaussian_NB = GaussianNB()
    try:
        Gaussian_NB.fit(x_train,y_train)
        Prediction = Gaussian_NB.predict(x_test)
    except:
        #warnings.warn('GaussianNaiveBayes algorithm cannot be performed with standard scalar!.',my_warning)
        print('GaussianNaiveBayes algorithm cannot be performed with standard scalar!.')
        return 0,0,0,0,0
    return metrics(y_test,Prediction)

def MultinomialNaiveBayes(x_train,x_test,y_train,y_test):
    Multinomial_NB = MultinomialNB()
    try:
        Multinomial_NB.fit(x_train, y_train)
        Prediction = Multinomial_NB.predict(x_test)         
    except:
        #warnings.warn('MultinomialNaiveBayes algorithm cannot be performed with standard scalar!.',my_warning)
        print('MultinomialNaiveBayes algorithm cannot be performed with standard scalar!.')
        return 0,0,0,0,0
    return metrics(y_test,Prediction)

def BernoulliNaiveBayes(x_train,x_test,y_train,y_test):
    Bernoulli_NB = BernoulliNB()
    try:
        Bernoulli_NB.fit(x_train,y_train)
        Prediction = Bernoulli_NB.predict(x_test)
    except:
        #warnings.warn('BernoulliNaiveBayes algorithm cannot be performed with standard scalar!.',my_warning)
        print('BernoulliNaiveBayes algorithm cannot be performed with standard scalar!.')
        return 0,0,0,0,0
    return metrics(y_test,Prediction)
