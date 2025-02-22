import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.drop(columns=['Serial No.'])  # Remove unnecessary column
    df.columns = [col.replace(' ', '_') for col in df.columns]  # Standardize column names
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    df[['GRE_Score', 'TOEFL_Score', 'CGPA']] = scaler.fit_transform(df[['GRE_Score', 'TOEFL_Score', 'CGPA']])
    
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data("../data/Admission_Predict.csv", "../outputs/cleaned_data.csv")