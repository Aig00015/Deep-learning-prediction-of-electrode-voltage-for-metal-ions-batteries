from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

svm_model = SVR(kernel='rbf') 


svm_model.fit(x_train.astype(float).values, y_train.values.ravel())