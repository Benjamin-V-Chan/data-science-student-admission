import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(input_path, output_path):
    df = pd.read_csv(input_path)
    print(df.describe())  # Summary statistics
    
    # Visualizing distributions
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Chance_of_Admit'], kde=True)
    plt.title('Distribution of Admission Chances')
    plt.savefig(f"{output_path}/admission_distribution.png")
    
    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlations')
    plt.savefig(f"{output_path}/correlation_heatmap.png")

if __name__ == "__main__":
    explore_data("../outputs/cleaned_data.csv", "../outputs")