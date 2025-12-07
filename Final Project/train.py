from project_imports import *
from prepare_ml_data import prepare_ml_data

train_path = "train_processed.csv"
test_path = "test_processed.csv"
X_train, X_test, y_train, y_test, X_Test = prepare_ml_data(train_path, test_path)

# LINEAR REGRESSION MODEL

lr = MultiOutputRegressor(LinearRegression(n_jobs=1))

# Fitting
lr = lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# RANDOMFOREST REGRESSOR MODEL

forest = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=1))

# Fitting
forest = forest.fit(X_train, y_train)
y_train_pred2 = forest.predict(X_train)
y_test_pred2 = forest.predict(X_test)

# GRADIENTBOOSTING REGRESSOR MODEL

bosting = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))

# Fitting
bosting = bosting.fit(X_train, y_train)
y_train_pred3 = bosting.predict(X_train)
y_test_pred3 = bosting.predict(X_test)

# KNEIGHBORS REGRESSOR MODEL

knn = MultiOutputRegressor(KNeighborsRegressor())

# Fitting
knn = knn.fit(X_train, y_train)
y_train_pred4 = knn.predict(X_train)
y_test_pred4 = knn.predict(X_test)

# RIDGE REGRESSION MODEL

reg = MultiOutputRegressor(linear_model.Ridge())

# Fitting
knn = reg.fit(X_train, y_train)
y_train_pred5 = reg.predict(X_train)
y_test_pred5 = reg.predict(X_test)

# DECISIONTREE REGRESSOR MODEL

dt = MultiOutputRegressor(DecisionTreeRegressor(max_depth=50, random_state=1))

# Fitting
dt = dt.fit(X_train, y_train)
y_train_pred6 = dt.predict(X_train)
y_test_pred6 = dt.predict(X_test)

# EVALUATION FOR LINEAR REGRESSION

print("Mean Square Error on training Data:{}".format(mean_squared_error(y_train, y_train_pred)))
print("Mean Square Error on testing Data:{}".format(mean_squared_error(y_test, y_test_pred)))
print("R2 score train:{}".format(r2_score(y_train, y_train_pred)))
print("R2 score test:{}".format(r2_score(y_test, y_test_pred)))
y_Test_pred = lr.predict(X_Test)

# EVALUATION FOR RANDOMFOREST REGRESSOR

print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred2)))
print("MSE test;{}".format(mean_squared_error(y_test, y_test_pred2)))
print("R2 score train:{}".format(r2_score(y_train, y_train_pred2)))
print("R2 score test:{}".format(r2_score(y_test, y_test_pred2)))

y_Test_pred2 = forest.predict(X_Test)
y_Test_pred2[2]

# EVALUATION FOR GRADIENTBOOSTING REGRESSOR

print("Mean Square Error on training Data:{}".format(mean_squared_error(y_train, y_train_pred3)))
print("Mean Square Error on testing Data:{}".format(mean_squared_error(y_test, y_test_pred3)))
print("R2 score train:{}".format(r2_score(y_train, y_train_pred3)))
print("R2 score test:{}".format(r2_score(y_test, y_test_pred3)))
y_Test_pred3 = bosting.predict(X_Test)
y_Test_pred3[3]

# EVALUATION FOR KNEIGHBORS REGRESSOR

print("Mean Square Error on training Data:{}".format(mean_squared_error(y_train, y_train_pred4)))
print("Mean Square Error on testing Data:{}".format(mean_squared_error(y_test, y_test_pred4)))
print("R2 score train:{}".format(r2_score(y_train, y_train_pred4)))
print("R2 score test:{}".format(r2_score(y_test, y_test_pred4)))
y_Test_pred4 = bosting.predict(X_Test)
y_Test_pred4[4]

# EVALUATION FOR RIDGE REGRESSION

print("Mean Square Error on training Data:{}".format(mean_squared_error(y_train, y_train_pred5)))
print("Mean Square Error on testing Data:{}".format(mean_squared_error(y_test, y_test_pred5)))
print("R2 score train:{}".format(r2_score(y_train, y_train_pred5)))
print("R2 score test:{}".format(r2_score(y_test, y_test_pred5)))
y_Test_pred5 = reg.predict(X_Test)
y_Test_pred5[5]

# EVALUATION FOR DECISIONTREE REGRESSOR

print("Mean Square Error on training Data:{}".format(mean_squared_error(y_train, y_train_pred6)))
print("Mean Square Error on testing Data:{}".format(mean_squared_error(y_test, y_test_pred6)))
print("R2 score train:{}".format(r2_score(y_train, y_train_pred6)))
print("R2 score test:{}".format(r2_score(y_test, y_test_pred6)))
y_Test_pred6 = dt.predict(X_Test)
y_Test_pred6[5]