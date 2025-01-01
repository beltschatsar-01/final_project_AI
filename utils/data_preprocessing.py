import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df, output_path):
    """
    Preprocess raw data: handle missing values, scale numerical features, and save the processed data.

    Args:
        df (pd.DataFrame): Raw data as a DataFrame.
        output_path (str): Path to save the processed data file (CSV).

    Returns:
        pd.DataFrame: Processed data ready for modeling.
    """
    try:
        if df is None or df.empty:
            print("The input DataFrame is empty or None.")
            return None
        print("Raw data loaded successfully!")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("Missing values handled.")

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    print("Categorical variables encoded.")

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print("Numerical features standardized.")

    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")

    return df

if __name__ == "__main__":
    raw_data_path = "../data/raw_data.csv"  
    processed_data_path = "../data/processed_data.csv"  

    df = pd.read_csv(raw_data_path)  
    processed_data = preprocess_data(df, processed_data_path)
