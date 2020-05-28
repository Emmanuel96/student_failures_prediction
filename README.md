# Student_failures_prediction

Create a regression model to predict students number of failures with the provided dataset.

## Objective

## Requirements

1. Python 3.7 or any working version

2. VS Code or Spyder

## Implementation

Once our environment is set up, we first import our Panda Libraries as follow:

    import pandas as pd
    import numpy as np
    import seaborn as sns

Next, we import the necessary scikit-learn libraries:

    # sklearn imports
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn import metrics

    # Regression libs
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet
    from sklearn.linear_model import OrthogonalMatchingPursuitCV
    from sklearn.preprocessing import LabelEncoder
    from sklearn.neural_network import MLPRegressor

### Regression

Firstly, we read our csv file,create a data frame out of it and drop our null values as follows:

    data = pd.read_csv(
        r'C:/Users/Emmanuel/Documents/projects/Python/Students Data Analysis/Dataset/student.csv')
    student_data = pd.DataFrame(data)

    # drop all null data
    student_data.dropna(inplace=True)

#### Handle Categorical Values

We handle our categorical data as follows:

    # array of categorical features
    cat_data = ['school', 'sex', 'address', 'famsize', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'activities', 'nursery', 'fatherd', 'Pstatus', 'higher', 'internet', 'romantic', 'famrel',
                'freetime', 'goout', 'Dalc', 'Walc', 'health', 'Medu', 'famsup']

    # convert categorical data to dummy variables
    student_data = handle_cat_data(cat_data, student_data)

#### Split Dataset To Test and Train Data

We can't use our entire dataset directly for both training and testing, hence we split it into 80% for training and 20% for testing with Pythons train_test_split method. We select our higher column as our target column and leave the other columns, except the target column to train our model with. We achieve this with the code snippet below:

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

## Results

The table below shows the accuracy of the different algorithms used. These values may vary slightly from what you get based on different factors i.e. your machine.

| Classifier        | Accuracy |
| ----------------- | -------- |
| Ridge Regression  | 22%      |
| MLP Regressor     | 19%      |
| Linear Regression | 22%      |

Welp! The accuracy ended up not being so good, but I guess that's okay because it means we just have to investigate more algorithms and pre processing functions.

---- THANK YOU FOR GOING THROUGH MY WORK -------
