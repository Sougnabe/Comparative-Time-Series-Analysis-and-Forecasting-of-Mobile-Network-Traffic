import matplotlib.pyplot as plt
import seaborn as sns

def plot_pdf_total(df, value_col='CDR'):
    total = df.groupby('Square id')[value_col].sum().reset_index()
    plt.figure(figsize=(8,5))
    sns.histplot(total[value_col], bins=200, kde=True)
    plt.title('PDF of total traffic per grid square')
    plt.xlabel('Total traffic')
    plt.ylabel('Density')
    plt.tight_layout()
