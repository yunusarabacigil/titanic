import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def build_model(input_path):
    titanic = pd.read_csv('../data/processed/titanic_processed.csv')
    
    # Preprocess the dataset
    # Drop rows with missing values
    titanic.dropna(subset=['age', 'embarked', 'deck', 'embark_town', 'alive'], inplace=True)
    
    # Convert categorical variables into dummy/indicator variables
    titanic = pd.get_dummies(titanic, columns=['sex', 'embarked', 'class', 'who', 'deck', 'embark_town', 'alive'], drop_first=True)
    
    # Define the feature matrix (X) and the target vector (y)
    X = titanic.drop(['survived', 'alive_yes'], axis=1)
    y = titanic['survived']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Print the classification report
    print(classification_report(y_test, y_pred))
    
    print("Model building completed.")
    
if __name__ == "__main__":
    build_model('../data/processed/titanic_processed.csv')
    
''' ///   # Function to predict survival given input data
def predict_survival(model, input_data):
    # Preprocess the input data similar to the training data
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    
    # Align input_df with the training dataset's columns
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_df)
    return prediction[0]

# Example input data
example_input = {
    'pclass': 3,
    'age': 25,
    'sibsp': 0,
    'parch': 0,
    'fare': 7.25,
    'sex_male': 1,
    'embarked_Q': 0,
    'embarked_S': 1,
    'class_Second': 0,
    'class_Third': 1,
    'who_man': 1,
    'who_woman': 0,
    'deck_B': 0,
    'deck_C': 0,
    'deck_D': 0,
    'deck_E': 0,
    'deck_F': 0,
    'deck_G': 0,
    'embark_town_Cherbourg': 0,
    'embark_town_Queenstown': 0,
    'embark_town_Southampton': 1,
    'alive_no': 0 }

# Predict survival for the example input data
 prediction = predict_survival(model, example_input)
 print(f'Prediction for the example input: {"Survived" if prediction == 1 else "Did not survive"}')       /// '''