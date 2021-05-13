<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<img src="https://img.shields.io/github/contributors/anand-gokul/pyevals?style=flat"/>
<img src="https://img.shields.io/github/languages/code-size/anand-gokul/pyevals?style=flat"/>
<img src="https://img.shields.io/github/license/anand-gokul/pyevals?style=flat"/>
<img src="https://img.shields.io/github/stars/anand-gokul/pyevals?style=flat"/>
<img src="https://img.shields.io/github/languages/top/anand-gokul/pyevals"/>


<!-- TABLE OF CONTENTS -->
## Table of Contents


* [About the Project](#about-the-project)
* [Installation](#installation)
* [Usage](#usage)
* [Future Work](#Futurework)
* [Contact](#contact)


<!-- ABOUT THE PROJECT -->
## About The Project


A very elegant and simple library to evaluate models.

This  module builds the BarPlot, BoxPlot, CountPlot, DistPlot, HeatMap, PairPlot and ViolinPlot only with one line of code.A folder is created 'Plots' where the pdf files of all the plots are stored.Along with this, a pdf file will be generated 'FinalPlots.pdf' which contains all the plots with which EDA can be performed easily.

This module will evaulate the Classification problems and Regression problems with 12 and 7 algorithms respectively. 

The Classification algorithms are KNN,LogisticRegression,DecisionTreeClassifier, RandomForestClassifier, SupportVectorClassifier, QuadraticDiscriminantSnalysis, SGDClassifier, AdaBoost, CalibratedClassifier, MultinomialNB, BernoulliNB, GaussianNB.

The Regression algorithms are LinearRegression, PolynomialRegression, RidgeRegression, LassoRegression, SupportVectorRegressor, GradientBoostingRegression, PLSRegression.

We also have implmented the Adjusted R Squared method as the Regression Metric Evaluation.

In Classification , Highest Accuracy is Highlighted in Yellow colour.

In Regression Model , Least Error is Highlighted in Yellow colour.


## Installation


1. Clone the repo
```sh
git clone https://github.com/Anand-gokul/pyevals.git
```

2. Install using pip or pip3
```commandline

pip3 install pyevals

(or)

pip install pyevals

```

<!-- USAGE EXAMPLES -->
## Usage


```python
import pyevals

# For Exploratory Data Analysis (or) For building the plots

pyevals.BuildPlots(data,CategoricalFeatures,ContinuousFeatures)

'''CategoricalFeatures and the ContinuousFeatures are the lists of the Categorical
and Continuous Features of the dataset respectively. '''


# For Classification

Object = pyevals.build(x_train,x_test,y_train,y_test,'classification')
Object.evaluate()

# For Regression

Object = pyevals.build(x_train,x_test,y_train,y_test,'regression')
Object.evaluate()

  
```

## Future Work


In this version we are only providing the reports and the plots as many as possible.We are working on improviing the plots for better EDA.We will try to implement hyperparameter optimization techniques to get the better results. We will also try to implement other algorithms in classification and regression soon. 

<!-- CONTACT -->
## Contact


Sai Gokul Krishna Reddy Talla - [@Krish](https://www.linkedin.com/in/gokul-talla) - gokulkrishna.talla@gmail.com

Ananda Datta Sai Phanindra Tangirala - [@Anand](https://www.linkedin.com/in/ananda-datta-sai-phanindra-tangirala-62a4b5185) - tangiralaphanindra@gmail.com

Anirudh Palaparthi - [@anirudh8889](https://twitter.com/anirudh8889) - aniruddhapnbb@gmail.com

Project Link: [https://github.com/Anand-Gokul/pyevals](https://github.com/Anand-gokul/pyevals)
