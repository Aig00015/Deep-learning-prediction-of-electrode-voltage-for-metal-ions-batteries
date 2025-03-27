from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


krr_model = KernelRidge(kernel='rbf', alpha=1.0)  

krr_model.fit(x_train.astype(float).values, y_train.values.ravel())  