# Supervised-Machine-Learning
LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API. This project will use an entire year's worth of data (2019) to predict the credit risk of loans from the first quarter of the next year (2020)
- ### Convert categorical data to numeric and separate target feature for training and testing data

  ```python
    x_train = train_df.drop(['loan_status'], axis = 1)
    x_train = pd.get_dummies(x_train)
    y_train = train_df['loan_status']
    
    x_test = test_df.drop(['loan_status'], axis = 1)
    x_test = pd.get_dummies(x_test)
    y_test = test_df['loan_status']
  ``
- ### Train the Logistic Regression model on the unscaled data and print the model score

  ```python
    from sklearn.linear_model import LogisticRegression
    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)
    classifier = LogisticRegression()
    model = classifier.fit(x_train,y_train)
    model.score(x_test,y_test)

    print(f"Training Data Score: {model.score(x_train,y_train)}")
    print(f"Testing Data Score: {model.score(x_test,y_test)}")
  ```
- ### Train a Random Forest Classifier model and print the model score

  ```python
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(random_state=1, n_estimators=500).fit(x_train, y_train)

    print(f'Training Score: {clf.score(x_train, y_train)}')
    print(f'Testing Score: {clf.score(x_test, y_test)}')
  ```
- ### Scale data with StandardScaler

  ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(x_train)
    X_train_scaled = scaler.transform(x_train)
    X_test_scaled = scaler.transform(x_test)
  ```
- ### Revisit the Preprocessing with the scaled data
  
  ```python
    classifier.fit(X_train_scaled,y_train)
    classifier.score(x_test,y_test)

    print(f"Training Data Score: {classifier.score(X_train_scaled,y_train)}")
    print(f"Testing Data Score: {classifier.score(X_test_scaled,y_test)}")
    
    clf_scaled = RandomForestClassifier(random_state=1, n_estimators=500).fit(X_train_scaled, y_train)

    print(f"Training Data Score: {clf_scaled.score(X_train_scaled,y_train)}")
    print(f"Testing Data Score: {clf_scaled.score(X_test_scaled,y_test)}")
  ```
