from sklearn.kernel_ridge import KernelRidge


krr_model = KernelRidge(kernel='rbf', alpha=1.0)  

krr_model.fit(x_train.astype(float).values, y_train.values.ravel())  
