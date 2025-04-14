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
    data_loader.inspect_data(df)
    
    # Drop the identifier column if present
    if 'Student_ID' in df.columns:
        df.drop(columns=['Student_ID'], inplace=True)
    
    # ---------------------------
    # 2. Exploratory Data Analysis (EDA)
    # ---------------------------
    # Plot a correlation heatmap for numerical features.
    visualization.plot_correlation_heatmap(df)
    
    # You might also want to visualize missing values or distributions of key features
    # (This example uses the histogram of Age and boxplot of Starting_Salary)
    visualization.plot_histogram(df, 'Age')
    visualization.plot_boxplot(df, 'Starting_Salary')
    
    # ---------------------------
    # 3. Preprocess the Data
    # ---------------------------
    # Handle missing values using either drop or fill strategy.
    df_clean = preprocessing.handle_missing_values(df, strategy='drop')
    
    # Encode categorical variables (e.g., Gender, Field_of_Study, Current_Job_Level, Entrepreneurship)
    df_encoded = preprocessing.encode_categorical(df_clean)
    
    # Define numerical columns based on your dataset.
    numerical_columns = [
        'Age', 'High_School_GPA', 'SAT_Score', 'University_Ranking',
        'University_GPA', 'Internships_Completed', 'Projects_Completed',
        'Certifications', 'Soft_Skills_Score', 'Networking_Score',
        'Job_Offers', 'Starting_Salary', 'Career_Satisfaction',
        'Years_to_Promotion', 'Work_Life_Balance'
    ]
    
    # Scale the numerical features.
    df_encoded = preprocessing.scale_features(df_encoded, numerical_columns)
    
    # Remove outliers for the regression target (Starting_Salary).
    df_encoded = preprocessing.remove_outliers(df_encoded, column='Starting_Salary')
    
    # ---------------------------
    # 4. Model Training and Evaluation
    # ---------------------------
    # (A) Classification Task: Predicting Entrepreneurship (assumes binary outcome)
    if 'Entrepreneurship' in df_encoded.columns:
        X_class = df_encoded.drop('Entrepreneurship', axis=1)
        y_class = df_encoded['Entrepreneurship']
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
            X_class, y_class, test_size=0.3, random_state=42
        )
        
        # Baseline Model: Naive Bayes
        nb_model = models.train_naive_bayes(X_train_class, y_train_class)
        y_pred_nb = nb_model.predict(X_test_class)
        print("\n--- Naive Bayes Classification Evaluation ---")
        evaluation.evaluate_classification(y_test_class, y_pred_nb)
        
        # Optimized Model: Tune Random Forest Classifier
        tuned_rf_classifier = models.tune_random_forest_classifier(X_train_class, y_train_class)
        y_pred_rf = tuned_rf_classifier.predict(X_test_class)
        print("\n--- Tuned Random Forest Classifier Evaluation ---")
        evaluation.evaluate_classification(y_test_class, y_pred_rf)
    
    # (B) Regression Task: Predicting Starting_Salary
    if 'Starting_Salary' in df_encoded.columns:
        X_reg = df_encoded.drop('Starting_Salary', axis=1)
        y_reg = df_encoded['Starting_Salary']
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.3, random_state=42
        )
        
        # Baseline Model: Linear Regression
        linreg_model = models.train_linear_regression(X_train_reg, y_train_reg)
        y_pred_lr = linreg_model.predict(X_test_reg)
        print("\n--- Linear Regression Evaluation ---")
        rmse_lr, r2_lr = evaluation.evaluate_regression(y_test_reg, y_pred_lr)
        
        # Optimized Model: Tune Random Forest Regressor
        tuned_rf_regressor = models.tune_random_forest_regressor(X_train_reg, y_train_reg)
        y_pred_rf_reg = tuned_rf_regressor.predict(X_test_reg)
        print("\n--- Tuned Random Forest Regressor Evaluation ---")
        rmse_rf, r2_rf = evaluation.evaluate_regression(y_test_reg, y_pred_rf_reg)
        
        # Additional Optimized Model: Tune SVR
        tuned_svr = models.tune_svr(X_train_reg, y_train_reg)
        y_pred_svr = tuned_svr.predict(X_test_reg)
        print("\n--- Tuned SVR Evaluation ---")
        rmse_svr, r2_svr = evaluation.evaluate_regression(y_test_reg, y_pred_svr)
    
    # ---------------------------
    # 5. Model Comparison Visualization
    # ---------------------------
    # Compare regression models if the evaluations were computed.
    try:
        models_names = ['Linear Regression', 'Tuned RF', 'Tuned SVR']
        rmse_values = [rmse_lr, rmse_rf, rmse_svr]
        visualization.plot_bar_chart(models_names, rmse_values)
    except NameError:
        print("Some regression models were not evaluated; please check your regression pipeline.")

if __name__ == "__main__":
    main()
