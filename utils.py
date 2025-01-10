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