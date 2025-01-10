import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_dataset(data_path: str):
    return pd.read_csv(data_path)

def plot_item_distribution(dataset: pd.DataFrame, column_name: str):
    item_count = dataset[column_name].value_counts()
    plt.figure(figsize=(12, 8))
    sns.barplot(x=item_count.values, y=item_count.index, order=item_count.index)
    plt.xlabel('Number of Purchases')
    plt.ylabel(column_name)
    plt.show()


def plot_numerical_distribution(dataset: pd.DataFrame, column_name: str):
    plt.figure(figsize=(12, 8))
    sns.histplot(dataset[column_name], kde=True, color='skyblue', bins=30)
    plt.xlabel(column_name, fontsize=14)
    plt.ylabel('Number of purchases', fontsize=14)
    plt.title(f'Distribution of {column_name}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    

def plot_numerical_distribution_grid(dataset: pd.DataFrame, columns: list):
    num_columns = 3
    num_rows = (len(columns) + num_columns - 1) // num_columns
    fig, ax = plt.subplots(num_rows, num_columns, figsize=(20, 6 * num_rows))
    ax = ax.flatten()
    
    for i, column in enumerate(columns):
        sns.histplot(dataset[column], kde=True, color='skyblue', bins=30, ax=ax[i])
        ax[i].set_xlabel(column, fontsize=14)
        ax[i].set_ylabel('Number of purchases', fontsize=14)
        ax[i].set_title(f'Distribution of {column}', fontsize=16)
        ax[i].grid(True, linestyle='--', alpha=0.7)
        ax[i].tick_params(axis='x', labelsize=12)
        ax[i].tick_params(axis='y', labelsize=12)
    
    for j in range(i + 1, len(ax)):
        fig.delaxes(ax[j])
    
    plt.tight_layout()
    plt.show()


def plot_pie_chart(dataset: pd.DataFrame, column_name: str):
    counting = dataset[column_name].value_counts()
    counting.plot(kind='pie', autopct='%1.1f%%', startangle=90, legend=True)
    plt.ylabel('')
    plt.title(f'{column_name} Distribution')
    plt.show()
    
    
def plot_correlation_matrix(dataset: pd.DataFrame):
    plt.figure(figsize=(14, 10))
    sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', linewidths=.5)
    plt.title('Correlation Matrix')
    plt.show()