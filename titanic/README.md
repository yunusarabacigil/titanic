Titanic Classification Project


This project demonstrates a complete data science workflow using the Titanic dataset. The workflow includes data cleaning, exploratory data analysis (EDA), model building, and visualization.

Steps Completed:

1-Data Loading and Preprocessing:

-Loaded the Titanic dataset using Seaborn.
-Preprocessed the dataset by dropping rows with missing values and converting categorical variables into dummy/indicator variables.

2-Feature and Target Definition:

-Defined the feature matrix (X) and the target vector (y).

3-Data Splitting:

-Split the dataset into training and testing sets using train_test_split.

4-Model Training:

-Trained a RandomForestClassifier on the training data.

5-Model Evaluation:

-Made predictions on the test set and evaluated the model's performance using a classification report.

6-Survival Prediction Function:

-Created a function to predict survival given input data.

7-Data Visualization:

-Implemented various data visualizations using Seaborn, including PairGrid with KDE and scatter plots, Catplot with Swarm Plot and Violin Plot, and FacetGrid with multiple plots.




## Project Structure


iris-classification/
├── README.md
├── data/
│   ├── raw/
│   │   └── iris.csv
│   └── processed/
│       └── iris_processed.csv
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_model_building.ipynb
│   └── 04_visualization.ipynb
├── scripts/
│   ├── data_cleaning.py
│   ├── EDA.py
│   ├── model_building.py
│   └── visualization.py
├── results/
│   ├── results_summary.md
│   └── figures/
└── LICENSE



## How to Run

1. Data Cleaning:
   - Run `scripts/data_cleaning.py` or `notebooks/01_data_cleaning.ipynb`.
   
2. Exploratory Data Analysis (EDA):
   - Run `scripts/EDA.py` or `notebooks/02_EDA.ipynb`.
   
3. Model Building:
   - Run `scripts/model_building.py` or `notebooks/03_model_building.ipynb`.
   
4. Visualization:
   - Run `scripts/visualization.py` or `notebooks/04_visualization.ipynb`.

## Requirements

- Python 3.6+
- pandas
- seaborn
- matplotlib
- scikit-learn
