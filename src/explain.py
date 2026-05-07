import shap
import pandas as pd
import mlflow.sklearn
from preprocess import load_and_clean_data, feature_engineering

def explain_model():
    # 1. Load data and a trained model
    df = load_and_clean_data('data/Telco-Customer-Churn.csv')
    df = feature_engineering(df)
    X = df.drop('Churn', axis=1)
    
    # Load the winner (e.g., XGBoost) from your local MLflow runs
    # You can find the 'logged_model' path in the MLflow UI
    # For now, we'll use the one we just trained in the script
    from xgboost import XGBClassifier
    model = XGBClassifier().fit(X, df['Churn']) # Simplified for explanation

    # 2. Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 3. Global Importance Plot
    print("Generating Summary Plot...")
    shap.summary_plot(shap_values, X, show=False)
    import matplotlib.pyplot as plt
    plt.savefig('notebooks/shap_summary.png')
    print("Summary plot saved to notebooks/shap_summary.png")

if __name__ == "__main__":
    explain_model()