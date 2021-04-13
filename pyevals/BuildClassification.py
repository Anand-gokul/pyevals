import warnings
import numpy as np
from utils import *
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
from sklearn.metrics._classification import *
from exceptions import *


def metrics(actual, predicted, labels=None):
    """

    :param actual:
    :param predicted:
    :param labels:
    :return:
    """
    y_type, y_test, predicted = check_targets(actual, predicted)
    if y_type not in ("binary", "multiclass"):
        raise PyEvalsTypeError("Actual {0} and predicted {1} outputs shape is not the same.".
                               format(actual.shape, predicted.shape))
    if labels is None:
        labels = unique_labels(actual, predicted)
    else:
        labels = np.asarray(labels)
        labels_size = labels.size
        if labels_size == 0:
            raise PyEvalsValueError("Labels should not be empty.")
    AccuracyScore = accuracy_score(y_test,predicted)
    RecallScore = recall_score(y_test,predicted)
    PrecisionScore = precision_score(y_test,predicted)
    RocAucScore = roc_auc_score(y_test,predicted)
    f1Score = f1_score(y_test,predicted)

    return AccuracyScore,RecallScore,PrecisionScore,RocAucScore,f1Score


def KNearestNeighbor(X_train,X_test,y_train,y_test=None):
    """

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    if y_test is not None:
        kNN = KNeighborsClassifier()
        kNN = kNN.fit(X_train,y_train)
        predicted = kNN.predict(X_test)
        return metrics(y_test, predicted)
    else:
        kNN = KNeighborsClassifier()
        kNN = kNN.fit(X_train, y_train)
        predicted = kNN.predict(X_test)
        return predicted


def LogisticReg(X_train,X_test,y_train,y_test=None):
    """

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    if y_test is not None:
        Logistic_Regression = LogisticRegression()
        Logistic_Regression = Logistic_Regression.fit(X_train,y_train)
        predicted = Logistic_Regression.predict(X_test)
        return metrics(y_test,predicted)
    else:
        Logistic_Regression = LogisticRegression()
        Logistic_Regression = Logistic_Regression.fit(X_train, y_train)
        predicted = Logistic_Regression.predict(X_test)
        return predicted


def DecisionTree(X_train,X_test,y_train,y_test=None):
    """

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    if y_test is not None:
        DecisionTree_Classifier = DecisionTreeClassifier()
        DecisionTree_Classifier = DecisionTree_Classifier.fit(X_train,y_train)
        predicted = DecisionTree_Classifier.predict(X_test)
        return metrics(y_test,predicted)
    else:
        DecisionTree_Classifier = DecisionTreeClassifier()
        DecisionTree_Classifier = DecisionTree_Classifier.fit(X_train, y_train)
        predicted = DecisionTree_Classifier.predict(X_test)
        return predicted


def RandomForest(X_train,X_test,y_train,y_test=None):
    """

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    if y_test is not None:
        RandomForest_Classifier = RandomForestClassifier()
        RandomForest_Classifier = RandomForest_Classifier.fit(X_train,y_train)
        predicted = RandomForest_Classifier.predict(X_test)
        return metrics(y_test,predicted)
    else:
        RandomForest_Classifier = RandomForestClassifier()
        RandomForest_Classifier = RandomForest_Classifier.fit(X_train, y_train)
        predicted = RandomForest_Classifier.predict(X_test)
        return predicted


def SupportVector(X_train,X_test,y_train,y_test=None):
    """

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    if y_test is not None:
        SupportVectortClassifier = make_pipeline(SVC())
        SupportVectortClassifier.fit(X_train,y_train)
        predicted = SupportVectortClassifier.predict(X_test)
        return metrics(y_test,predicted)
    else:
        SupportVectortClassifier = make_pipeline(SVC())
        SupportVectortClassifier.fit(X_train, y_train)
        predicted = SupportVectortClassifier.predict(X_test)
        return predicted


def QuadraticDiscriminant(X_train,X_test,y_train,y_test=None):
    """

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    if y_test is not None:
        QuadraticDiscriminant_Analysis = QuadraticDiscriminantAnalysis()
        QuadraticDiscriminant_Analysis.fit(X_train,y_train)
        predicted = QuadraticDiscriminant_Analysis.predict(X_test)
        return metrics(y_test,predicted)
    else:
        QuadraticDiscriminant_Analysis = QuadraticDiscriminantAnalysis()
        QuadraticDiscriminant_Analysis.fit(X_train, y_train)
        predicted = QuadraticDiscriminant_Analysis.predict(X_test)
        return predicted


def SophisticatedGradientDescent(X_train,X_test,y_train,y_test=None):
    """

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    if y_test is not None:
        SGDClassifier_ = SGDClassifier()
        SGDClassifier_.fit(X_train,y_train)
        predicted = SGDClassifier_.predict(X_test)
        return metrics(y_test,predicted)
    else:
        SGDClassifier_ = SGDClassifier()
        SGDClassifier_.fit(X_train, y_train)
        predicted = SGDClassifier_.predict(X_test)
        return predicted


def AdaBoost(X_train,X_test,y_train,y_test=None):
    """

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    if y_test is not None:
        AdaBoostClassifier_ = AdaBoostClassifier()
        AdaBoostClassifier_.fit(X_train,y_train)
        predicted = AdaBoostClassifier_.predict(X_test)
        return metrics(y_test,predicted)
    else:
        AdaBoostClassifier_ = AdaBoostClassifier()
        AdaBoostClassifier_.fit(X_train, y_train)
        predicted = AdaBoostClassifier_.predict(X_test)
        return predicted


def CalibratedClassifier(X_train,X_test,y_train,y_test=None):
    """

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    if y_test is not None:
        CalibratedClassifierCV_ = CalibratedClassifierCV()
        CalibratedClassifierCV_ = CalibratedClassifierCV_.fit(X_train,y_train)
        predicted = CalibratedClassifierCV_.predict(X_test)
        return metrics(y_test,predicted)
    else:
        CalibratedClassifierCV_ = CalibratedClassifierCV()
        CalibratedClassifierCV_ = CalibratedClassifierCV_.fit(X_train, y_train)
        predicted = CalibratedClassifierCV_.predict(X_test)
        return predicted


def GaussianNaiveBayes(X_train,X_test,y_train,y_test=None):
    """

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    if y_test is not None:
        Gaussian_NB = GaussianNB()
        try:
            Gaussian_NB.fit(X_train,y_train)
            predicted = Gaussian_NB.predict(X_test)
        except:
            #warnings.warn('GaussianNaiveBayes algorithm cannot be performed with standard scalar!.',my_warning)
            print('GaussianNaiveBayes algorithm cannot be performed with standard scalar!.')
            return 0,0,0,0,0
        return metrics(y_test,predicted)
    else:
        Gaussian_NB = GaussianNB()
        try:
            Gaussian_NB.fit(X_train, y_train)
            predicted = Gaussian_NB.predict(X_test)
            return predicted
        except:
            # warnings.warn('GaussianNaiveBayes algorithm cannot be performed with standard scalar!.',my_warning)
            print('GaussianNaiveBayes algorithm cannot be performed with standard scalar!.')
            return 0, 0, 0, 0, 0


def MultinomialNaiveBayes(X_train,X_test,y_train,y_test=None):
    """

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    if y_test is not None:
        Multinomial_NB = MultinomialNB()
        try:
            Multinomial_NB.fit(X_train, y_train)
            predicted = Multinomial_NB.predict(X_test)
        except:
            #warnings.warn('MultinomialNaiveBayes algorithm cannot be performed with standard scalar!.',my_warning)
            print('MultinomialNaiveBayes algorithm cannot be performed with standard scalar!.')
            return 0,0,0,0,0
        return metrics(y_test,predicted)
    else:
        Multinomial_NB = MultinomialNB()
        try:
            Multinomial_NB.fit(X_train, y_train)
            predicted = Multinomial_NB.predict(X_test)
            return predicted
        except:
            # warnings.warn('MultinomialNaiveBayes algorithm cannot be performed with standard scalar!.',my_warning)
            print('MultinomialNaiveBayes algorithm cannot be performed with standard scalar!.')
            return 0, 0, 0, 0, 0


def BernoulliNaiveBayes(X_train,X_test,y_train,y_test=None):
    """

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    if y_test is not None:
        Bernoulli_NB = BernoulliNB()
        try:
            Bernoulli_NB.fit(X_train,y_train)
            predicted = Bernoulli_NB.predict(X_test)
        except:
            #warnings.warn('BernoulliNaiveBayes algorithm cannot be performed with standard scalar!.',my_warning)
            print('BernoulliNaiveBayes algorithm cannot be performed with standard scalar!.')
            return 0,0,0,0,0
        return metrics(y_test,predicted)
    else:
        Bernoulli_NB = BernoulliNB()
        try:
            Bernoulli_NB.fit(X_train, y_train)
            predicted = Bernoulli_NB.predict(X_test)
            return predicted
        except:
            # warnings.warn('BernoulliNaiveBayes algorithm cannot be performed with standard scalar!.',my_warning)
            print('BernoulliNaiveBayes algorithm cannot be performed with standard scalar!.')
            return 0, 0, 0, 0, 0