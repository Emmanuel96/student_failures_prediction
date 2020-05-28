import pandas as pd
import numpy as np
import seaborn as sns

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# sklearn imports: accuracy and error readings
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics

# sklearn split test and train lib
from sklearn.model_selection import train_test_split

# Regression libs
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor

# miscellaneous
import time


# function to run our regression models
def run_reg_models(classifer_names, classifiers, X_train, X_test, y_train, y_test):
    counter = 0
    for name, clf in zip(classifer_names, classifiers):
        result = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        model_performance = pd.DataFrame(data=[r2_score(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred))],
                                         index=["R2", "RMSE"])
        print(name + ' performance: ')
        print(model_performance)

# function to convert categorical data to dummy data


def handle_cat_data(cat_feats, data):
    for f in cat_feats:
        to_add = pd.get_dummies(data[f], prefix=f, drop_first=True)
        merged_list = data.join(
            to_add, how='left', lsuffix='_left', rsuffix='_right')
        data = merged_list

    # then drop the categorical features
    data.drop(cat_feats, axis=1, inplace=True)

    return data


# read csv file
data = pd.read_csv(
    'C:/Users/Emmanuel/Documents/projects/Python/Students Data Analysis/Dataset//student.csv')
student_data = pd.DataFrame(data)

# drop all null data
student_data.dropna(inplace=True)

# array of categorical features
cat_data = ['school', 'sex', 'address', 'famsize', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'activities', 'nursery', 'fatherd', 'Pstatus', 'higher', 'internet', 'romantic', 'famrel',
            'freetime', 'goout', 'Dalc', 'Walc', 'health', 'Medu', 'famsup']

# convert categorical data to dummy variables
student_data = handle_cat_data(cat_data, student_data)

# split testing and training data
X_train, X_test, y_train, y_test = train_test_split(student_data.drop(
    'failures', axis=1), student_data.failures, test_size=0.25, stratify=student_data.failures)


reg_algs_names = ['Linear Regression', 'Ridge Regression', 'Lasso Regression',
                  'Elastic Net Regression', 'Orthongonal Matching Pursuit CV',
                  'MLP Regressor']

reg_algs = [
    LinearRegression(normalize=True),
    Ridge(alpha=0, normalize=True),
    Lasso(alpha=0.01, normalize=False),
    ElasticNet(random_state=0),
    OrthogonalMatchingPursuitCV(cv=8, normalize=True),
    MLPRegressor(max_iter=1000)
]

run_reg_models(reg_algs_names, reg_algs, X_train, X_test, y_train, y_test)
