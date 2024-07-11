import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_visualizations(input_path):
    # Load the Titanic dataset
    titanic = pd.read_csv('../data/processed/titanic_processed.csv')

    # Create a PairGrid with KDE and scatter plots
    g = sns.PairGrid(titanic, vars=['age', 'fare', 'pclass'], hue='survived')
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot, cmap='Blues_d')
    g.map_diag(sns.histplot)
    g.add_legend()

    plt.savefig('../results/figures/pairgrid.png')
    plt.show()


    # Create a Catplot with Swarm Plot and Violin Plot
    sns.catplot(x='class', y='age', hue='survived', kind='violin', split=True, inner='quartile', data=titanic)
    sns.swarmplot(x='class', y='age', hue='survived', data=titanic, dodge=True, color='k', alpha=0.5)

    plt.savefig('../results/figures/swarm.png')
    plt.show()

    # Create a FacetGrid with multiple plots
    g = sns.FacetGrid(titanic, col='sex', row='class', margin_titles=True)
    g.map(sns.histplot, 'age', kde=True, bins=20)

    plt.savefig('../results/figures/facetGrid.png')
    plt.show()

if __name__ == "__main__":
    create_visualizations('../data/processed/titanic_processed.csv')