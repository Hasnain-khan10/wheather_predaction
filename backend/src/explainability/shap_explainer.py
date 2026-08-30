import shap
import pandas as pd
import numpy as np
from typing import Dict, Any

def compute_shap_values(model: Any, X: pd.DataFrame) -> Dict[str, float]:
    """
    Computes SHAP values using TreeExplainer for tree-based models
    and returns JSON-serializable feature importance values.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception:
        # Fallback for non-tree models
        background = shap.kmeans(X, 10)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X.sample(n=min(100, len(X))))
        
    if isinstance(shap_values, list):
        shap_values = shap_values[1] 
        
    # Calculate mean absolute SHAP values for feature importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    importance_dict = {col: float(val) for col, val in zip(X.columns, mean_abs_shap)}
    
    # Sort descending
    importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
    
    return importance_dict
