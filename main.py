# main.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import data_loader, preprocessing, visualization, models, evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def classify_target(X, y, target_name, class_labels):
    """
    Generic function to train and evaluate two classifiers for a given target.
    In addition to standard classification metrics, it computes RMSE and R^2
    by converting the categorical predictions to numeric codes (assuming an ordinal
    relationship among the classes). The function returns a dictionary of metrics.
    
    Parameters:
      X (DataFrame): Preprocessed feature set.
      y (Series): Target variable for classification.
      target_name (str): Name of the target (for logging purposes).
      class_labels (list): Ordered list of class labels.
    
    Returns:
      dict: A dictionary with keys 'NB' and 'GBC', each containing a sub-dictionary with 'rmse' and 'r2'.
    """
    print(f"\n=== Classification for Target: {target_name} ===")
    
    # Split the data for classification.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    metrics = {}
    
    # ----- Simple Algorithm: Gaussian Naive Bayes -----
    nb_model = models.train_naive_bayes(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    print(f"\n--- Gaussian Naive Bayes Evaluation for {target_name} ---")
    evaluation.evaluate_classification(y_test, y_pred_nb, class_labels=class_labels)
    
    # Convert the categorical results to numeric codes for RMSE and R^2 calculation.
    y_test_num = pd.Categorical(y_test, categories=class_labels, ordered=True).codes
    y_pred_nb_num = pd.Categorical(y_pred_nb, categories=class_labels, ordered=True).codes
    rmse_nb = np.sqrt(mean_squared_error(y_test_num, y_pred_nb_num))
    r2_nb = r2_score(y_test_num, y_pred_nb_num)
    print(f"RMSE (Gaussian NB): {rmse_nb:.4f}")
    print(f"R² (Gaussian NB): {r2_nb:.4f}")
    metrics['NB'] = {'rmse': rmse_nb, 'r2': r2_nb}
    
    # ----- Sophisticated Algorithm: Gradient Boosting Classifier -----
    gbc_model = models.train_gradient_boosting_classifier(X_train, y_train)
    y_pred_gbc = gbc_model.predict(X_test)
    print(f"\n--- Gradient Boosting Classification Evaluation for {target_name} ---")
    evaluation.evaluate_classification(y_test, y_pred_gbc, class_labels=class_labels)
    
    # Convert predictions to numeric codes.
    y_pred_gbc_num = pd.Categorical(y_pred_gbc, categories=class_labels, ordered=True).codes
    rmse_gbc = np.sqrt(mean_squared_error(y_test_num, y_pred_gbc_num))
    r2_gbc = r2_score(y_test_num, y_pred_gbc_num)
    print(f"RMSE (Gradient Boosting): {rmse_gbc:.4f}")
    print(f"R² (Gradient Boosting): {r2_gbc:.4f}")
    metrics['GBC'] = {'rmse': rmse_gbc, 'r2': r2_gbc}
    
    return metrics

def plot_model_comparison(metrics, target_name):
    """
    Create bar charts to compare RMSE and R^2 for two models.
    
    Parameters:
      metrics (dict): A dictionary containing keys 'NB' and 'GBC' with sub-dicts for 'rmse' and 'r2'.
      target_name (str): Name of the target to use in the title.
    """
    models_names = ['Gaussian NB', 'Gradient Boosting']
    rmse_values = [metrics['NB']['rmse'], metrics['GBC']['rmse']]
    r2_values = [metrics['NB']['r2'], metrics['GBC']['r2']]
    
    # Plot RMSE comparison
    plt.figure(figsize=(8, 5))
    plt.bar(models_names, rmse_values, color=['skyblue', 'salmon'], edgecolor='black')
    plt.title(f"RMSE Comparison for {target_name}")
    plt.ylabel("RMSE")
    plt.ylim(0, max(rmse_values)*1.2)
    plt.show()
    
    # Plot R² comparison
    plt.figure(figsize=(8, 5))
    plt.bar(models_names, r2_values, color=['skyblue', 'salmon'], edgecolor='black')
    plt.title(f"R² Comparison for {target_name}")
    plt.ylabel("R²")
    plt.ylim(min(r2_values)*1.2 if min(r2_values) < 0 else 0, 1)
    plt.show()

def main():
    # ---------------------------
    # 1. Load and Inspect Data
    # ---------------------------
    data_path = os.path.join('data', 'education_career_success.csv')
    df = data_loader.load_data(data_path)
    
    # Drop identifier columns (e.g., Student_ID)
    if 'Student_ID' in df.columns:
        df.drop(columns=['Student_ID'], inplace=True)
    
    print("=== Initial Data Inspection ===")
    preprocessing.inspect_data(df)
    
    print("=== Data Validation ===")
    preprocessing.validate_data(df)
    
    # ---------------------------
    # 2. Exploratory Data Analysis (EDA)
    # ---------------------------
    visualization.plot_correlation_heatmap(df)
    visualization.plot_histogram(df, 'Age')
    visualization.plot_boxplot(df, 'Starting_Salary')
    
    # ---------------------------
    # 3. Preprocess the Data for Modeling
    # ---------------------------
    df_clean = preprocessing.handle_missing_values(df, strategy='fill')
    df_clean = preprocessing.remove_outliers(df_clean, column='Starting_Salary')
    
    # ---------------------------
    # 4. Prepare Target and Features for Classification
    # ---------------------------
    if "Current_Job_Level" not in df_clean.columns:
        print("Error: 'Current_Job_Level' column not found!")
        return
    
    y_class = df_clean["Current_Job_Level"]
    df_features = df_clean.drop(columns=["Current_Job_Level"])
    
    # Encode categorical variables in features.
    df_encoded = preprocessing.encode_categorical(df_features)
    
    # Scale numerical features.
    numerical_columns = [
        'Age', 'High_School_GPA', 'SAT_Score', 'University_Ranking',
        'University_GPA', 'Internships_Completed', 'Projects_Completed',
        'Certifications', 'Soft_Skills_Score', 'Networking_Score',
        'Job_Offers', 'Starting_Salary', 'Career_Satisfaction',
        'Years_to_Promotion', 'Work_Life_Balance'
    ]
    numerical_columns = [col for col in numerical_columns if col not in ["Current_Job_Level"]]
    df_encoded = preprocessing.scale_features(df_encoded, numerical_columns, scaler_type='robust')
    
    # ---------------------------
    # 5. Classification Task on Current_Job_Level
    # ---------------------------
    class_labels = sorted(y_class.unique())
    metrics = classify_target(df_encoded, y_class, target_name="Current_Job_Level", class_labels=class_labels)
    
    # ---------------------------
    # 6. Visualization: Compare Model Performances
    # ---------------------------
    plot_model_comparison(metrics, target_name="Current_Job_Level")
    
    # ---------------------------
    # 7. (Optional) Regression Task: Baseline Mean Predictor for Starting_Salary
    # ---------------------------
    if "Starting_Salary" in df_clean.columns:
        y_reg = df_clean["Starting_Salary"]
        X_reg = df_clean.drop("Starting_Salary", axis=1)
        X_reg_encoded = preprocessing.encode_categorical(X_reg)
        
        numerical_columns_reg = [col for col in numerical_columns if col != "Starting_Salary"]
        X_reg_encoded = preprocessing.scale_features(X_reg_encoded, numerical_columns_reg, scaler_type='robust')
        
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg_encoded, y_reg, test_size=0.3, random_state=42
        )
        
        baseline_model = models.baseline_mean_regressor(X_train_reg, y_train_reg)
        y_pred_baseline = baseline_model.predict(X_test_reg)
        
        rmse_baseline = np.sqrt(mean_squared_error(y_test_reg, y_pred_baseline))
        r2_baseline = r2_score(y_test_reg, y_pred_baseline)
        print("\n--- Baseline Mean Predictor Evaluation (Regression) ---")
        print("Baseline RMSE:", rmse_baseline)
        print("Baseline R²:", r2_baseline)
    
if __name__ == "__main__":
    main()
