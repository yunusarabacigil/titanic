import os
import seaborn as sns
import pandas as pd

def clean_titanic_data(input_path, output_path):
    # Load Titanic dataset
    titanic = sns.load_dataset('titanic')
    
    # Check for null values
    print("Null values before cleaning:")
    print(titanic.isnull().sum())
    
    # Drop rows with null values
    titanic_clean = titanic.dropna()
    
    # Check again after cleaning
    print("\nNull values after cleaning:")
    print(titanic_clean.isnull().sum())
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save cleaned data to a new file
    titanic_clean.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to {output_path}")

if __name__ == "__main__":
    clean_titanic_data('titanic.csv', '../data/processed/titanic_processed.csv')