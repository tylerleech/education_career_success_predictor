# main.py
import os
from src import data_loader, preprocessing, visualization, models, evaluation
from sklearn.model_selection import train_test_split

def main():
    # ---------------------------
    # 1. Load and Inspect Data
    # ---------------------------
    data_path = os.path.join('data', 'education_career_success.csv')
    df = data_loader.load_data(data_path)
    
    # Initial inspection and validation of the raw data
    print("=== Initial Data Inspection ===")
    preprocessing.inspect_data(df)
    print("=== Data Validation ===")
    preprocessing.validate_data(df)
    
    # Drop any identifier columns that are not informative
    if 'Student_ID' in df.columns:
        df.drop(columns=['Student_ID'], inplace=True)
    
    # ---------------------------
    # 2. Exploratory Data Analysis (EDA)
    # ---------------------------
    visualization.plot_correlation_heatmap(df)
    visualization.plot_histogram(df, 'Age')
    visualization.plot_boxplot(df, 'Starting_Salary')
    
    # ---------------------------
    # 3. Preprocess the Data
    # ---------------------------
    # Handle missing values using a robust filling strategy
    df_clean = preprocessing.handle_missing_values(df, strategy='fill')
    
    # Remove outliers from key columns (e.g., the regression target)
    df_clean = preprocessing.remove_outliers(df_clean, column='Starting_Salary')
    
    # Encode categorical variables into numerical representations
    df_encoded = preprocessing.encode_categorical(df_clean)
    
    # Define numerical columns based on your dataset's features
    numerical_columns = [
        'Age', 'High_School_GPA', 'SAT_Score', 'University_Ranking',
        'University_GPA', 'Internships_Completed', 'Projects_Completed',
        'Certifications', 'Soft_Skills_Score', 'Networking_Score',
        'Job_Offers', 'Starting_Salary', 'Career_Satisfaction',
        'Years_to_Promotion', 'Work_Life_Balance'
    ]
    
    # Scale numerical features using a robust scaler to lessen the impact of outliers
    df_encoded = preprocessing.scale_features(df_encoded, numerical_columns, scaler_type='robust')
    
    # ---------------------------
    # 4. Model Training and Evaluation
    # ---------------------------
    
    # (A) Regression Task: Predicting Starting_Salary
    if 'Starting_Salary' in df_encoded.columns:
        X_reg = df_encoded.drop('Starting_Salary', axis=1)
        y_reg = df_encoded['Starting_Salary']
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.3, random_state=42
        )
        
        # Train the stacking regressor ensemble
        stacking_regressor_model = models.stacking_regressor(X_train_reg, y_train_reg)
        y_pred_stack = stacking_regressor_model.predict(X_test_reg)
        print("\n--- Stacking Regressor Evaluation ---")
        rmse_stack, r2_stack = evaluation.evaluate_regression(y_test_reg, y_pred_stack)
    
    # (B) Classification Task: Predicting Entrepreneurship (if this column exists)
    if 'Entrepreneurship' in df_encoded.columns:
        X_class = df_encoded.drop('Entrepreneurship', axis=1)
        y_class = df_encoded['Entrepreneurship']
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
            X_class, y_class, test_size=0.3, random_state=42
        )
        
        # Train the stacking classifier ensemble
        stacking_classifier_model = models.stacking_classifier(X_train_class, y_train_class)
        y_pred_class = stacking_classifier_model.predict(X_test_class)
        print("\n--- Stacking Classifier Evaluation ---")
        evaluation.evaluate_classification(y_test_class, y_pred_class)
    
    # ---------------------------
    # 5. Model Performance Summary (Optional Visualization)
    # ---------------------------
    try:
        print("\nStacking Regressor Performance:")
        print(f"RMSE: {rmse_stack:.4f}")
        print(f"R²: {r2_stack:.4f}")
    except NameError:
        print("Regression performance metrics are not available. Check your regression pipeline.")

if __name__ == "__main__":
    main()

