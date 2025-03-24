from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import torch

def evaluate_model(model, x_test, y_test, device):
    model.eval()
    x_tensor = torch.FloatTensor(x_test).to(device)
    with torch.no_grad():
        preds = model(x_tensor).cpu().numpy()
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
