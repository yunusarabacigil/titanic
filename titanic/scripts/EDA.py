import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def perform_eda(input_path):
    
    # Loading the Titanic Dataset
    titanic = pd.read_csv('../data/processed/titanic_processed.csv')
    
    # Displaying the First Few Rows
    print(titanic.head())
    
    # Displaying General Information About the Dataset
    print(titanic.info())
    print(titanic.describe())
    
    # Number of Passengers by Gender
    sns.countplot(data=titanic, x='sex')
    plt.title('Number of Passengers by Gender')
    plt.savefig('../results/figures/countPlot.png')
    plt.show()

    # Survival Rate by Class
    sns.countplot(data=titanic, x='class', hue='survived')
    plt.title('Survival Rate by Class')
    plt.savefig('../results/figures/countPlot1.png')
    plt.show()

    # Age Distribution
    sns.histplot(data=titanic, x='age', bins=20, kde=True)
    plt.title('Age Distribution')
    plt.savefig('../results/figures/histPlot.png')
    plt.show()

    # Fare Distribution
    sns.histplot(data=titanic, x='fare', bins=20, kde=True)
    plt.title('Fare Distribution')
    plt.savefig('../results/figures/histPlot1.png')
    plt.show()

    # Relationship Between Age and Fare
    sns.scatterplot(data=titanic, x='age', y='fare', hue='survived')
    plt.title('Relationship Between Age and Fare')
    plt.savefig('../results/figures/scatterPlot.png')
    plt.show()

    # Visualizing the Correlation Matrix (with just numerical data)
    numeric_titanic = titanic.select_dtypes(include=['float64', 'int64'])
    corr = numeric_titanic.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('../results/figures/heatMap.png')
    plt.show()

    # Age distribution by class
    sns.boxplot(data=titanic, x='class', y='age')
    plt.title('Age Distribution by Class')
    plt.savefig('../results/figures/boxPlot.png')
    plt.show()
    
    # Relationship Between Sex and Survival
    sns.barplot(data=titanic, x='sex', y='survived')
    plt.title('Relationship Between Sex and Survival')
    plt.savefig('../results/figures/barPlot.png')
    plt.show()
    
    # Relationship Between Sex and Survival by Class
    sns.catplot(data=titanic, x='sex', hue='survived', col='class', kind='count')
    plt.title('Relationship Between Sex and Survival by Class')
    plt.savefig('../results/figures/catPlot.png')
    plt.show()

    # Survival Rate by Embarked Port and Class
    sns.catplot(data=titanic, x='embarked', hue='survived', col='class', kind='count')
    plt.title('Survival Rate by Embarked Port and Class')
    plt.savefig('../results/figures/catPlot1.png')
    plt.show()
    
    # Relationship Between Class, Sex, and Survival
    sns.catplot(data=titanic, x='class', hue='survived', col='sex', kind='count')
    plt.title('Relationship Between Class, Sex, and Survival')
    plt.savefig('../results/figures/catPlot2.png')
    plt.show()

    # Relationship Between Age and Survival by Sex
    sns.violinplot(data=titanic, x='sex', y='age', hue='survived', split=True)
    plt.title('Relationship Between Age and Survival by Sex')
    plt.savefig('../results/figures/violinPlot.png')
    plt.show()
    
    print("EDA completed. All plot saved to results/figure")

if __name__ == "__main__":
    perform_eda('../data/processed/titanic_processed.csv')