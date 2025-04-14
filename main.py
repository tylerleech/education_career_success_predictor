import os
from src import data_loader, preprocessing, visualization, models, evaluation
from sklearn.model_selection import train_test_split

def main():
    # ---------------------------
    # 1. Load and Inspect Data
    # ---------------------------
    # Define the file path for your dataset.
    data_path = os.path.join('data', 'education_career_success.csv')
    
    # Load the data using the data_loader module.
    df = data_loader.load_data(data_path)
    data_loader.inspect_data(df)
    
    # Drop identifier column(s) that are not useful for modeling.
    if 'Student_ID' in df.columns:
        df.drop(columns=['Student_ID'], inplace=True)
    
    # ---------------------------
    # 2. Preprocess the Data
    # ---------------------------
    # Handle missing values. Strategy options: 'drop' or 'fill'. Here we drop rows with missing values.
    df_clean = preprocessing.handle_missing_values(df, strategy='drop')
    
    # Encode categorical variables.
    # This will convert columns such as Gender, Field_of_Study, Current_Job_Level, and Entrepreneurship
    # into dummy/indicator variables.
    df_encoded = preprocessing.encode_categorical(df_clean)
    
    # Define the numerical columns that need to be scaled.
    # These are the continuous features from your dataset.
    numerical_columns = [
        'Age', 'High_School_GPA', 'SAT_Score', 'University_Ranking',
        'University_GPA', 'Internships_Completed', 'Projects_Completed',
        'Certifications', 'Soft_Skills_Score', 'Networking_Score',
        'Job_Offers', 'Starting_Salary', 'Career_Satisfaction',
        'Years_to_Promotion', 'Work_Life_Balance'
    ]
    
    # Scale the numerical features.
    df_encoded = preprocessing.scale_features(df_encoded, numerical_columns)
    
    # Remove outliers for the regression target. Here we use 'Starting_Salary'.
    df_encoded = preprocessing.remove_outliers(df_encoded, column='Starting_Salary')
    
    # ---------------------------
    # 3. Visualize Data
    # ---------------------------
    # Plot a histogram for Age.
    visualization.plot_histogram(df_encoded, 'Age')
    # Plot a scatter chart, for example: SAT_Score vs. High_School_GPA.
    visualization.plot_scatter(df_encoded, 'SAT_Score', 'High_School_GPA')
    # Plot a boxplot for Starting_Salary to inspect its distribution and potential outliers.
    visualization.plot_boxplot(df_encoded, 'Starting_Salary')
    
    # ---------------------------
    # 4. Model Training and Evaluation
    # ---------------------------
    
    # 4.1 Classification Task: Predicting Entrepreneurship (binary Yes/No)
    if 'Entrepreneurship' in df_encoded.columns:
        # For classification, define X and y.
        X_class = df_encoded.drop('Entrepreneurship', axis=1)
        y_class = df_encoded['Entrepreneurship']
        
        # Split the data into training and testing sets.
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
            X_class, y_class, test_size=0.3, random_state=42
        )
    
        # Train a baseline classification model: Naive Bayes.
        nb_model = models.train_naive_bayes(X_train_class, y_train_class)
        y_pred_nb = nb_model.predict(X_test_class)
        print("Naive Bayes Classification Evaluation:")
        evaluation.evaluate_classification(y_test_class, y_pred_nb)
    
    # 4.2 Regression Task: Predicting Starting_Salary
    if 'Starting_Salary' in df_encoded.columns:
        # For regression, define X and y.
        X_reg = df_encoded.drop('Starting_Salary', axis=1)
        y_reg = df_encoded['Starting_Salary']
        
        # Split the data into training and testing sets.
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.3, random_state=42
        )
    
        # Train a baseline regression model: Linear Regression.
        linreg_model = models.train_linear_regression(X_train_reg, y_train_reg)
        y_pred_lr = linreg_model.predict(X_test_reg)
        print("Linear Regression Evaluation:")
        rmse_lr, r2_lr = evaluation.evaluate_regression(y_test_reg, y_pred_lr)
    
    # ---------------------------
    # 5. Model Comparison Visualization
    # ---------------------------
    # For demonstration, assume you have RMSE values for several regression models.
    # Here, we use placeholder RMSE values for Random Forest and SVR. Replace these with actual values as you add models.
    models_names = ['Linear Regression', 'Random Forest', 'SVR']
    rmse_values = [rmse_lr, 20.0, 25.0]  # Update 20.0 and 25.0 based on your model evaluations.
    visualization.plot_bar_chart(models_names, rmse_values)

if __name__ == "__main__":
    main()
