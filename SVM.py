from sklearn.svm import SVR

svm_model = SVR(kernel='rbf') 


svm_model.fit(x_train.astype(float).values, y_train.values.ravel())
