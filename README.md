# Credit Risk Prediction with supervised machine learning algorithms
## Objective 
LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API. This project will use an entire year's worth of data (2019) to predict the credit risk of loans from the first quarter of the next year (2020)

## Model included
- Logistic Regression
- Random Forest

## Process
- Data Preparetion
  - Convert categorical data to numeric and separate target feature for training and testing data
  - Train the Logistic Regression model on the unscaled data 

  ```python
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    model = classifier.fit(x_train,y_train)
    model.score(x_test,y_test)
  ```
  - Train a Random Forest Classifier model and print the model score

  ```python
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(random_state=1, n_estimators=500).fit(x_train, y_train)
  ```
  - Scale data with StandardScaler

  ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(x_train)
    X_train_scaled = scaler.transform(x_train)
    X_test_scaled = scaler.transform(x_test)
  ```
  - Fit the model with the scaled data

## Summary
Scaling data is important before fit the model, different from normalization, StandardScaler scales each input variable separately by shifting the distribution to have a mean of zero and a standard deviation of one. StandardScaler is useful especially for classification model. In this case, after scaling the data, the model score for the training data set increased from 0.648 to 0.713, and the model score for the testing data set increased from  0.5253 to 0.72.
