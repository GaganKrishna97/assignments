import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def auto_eda_report(csv_path, output_dir='eda_report'):
    # Load dataset
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # Strip whitespace from column names

    # Make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    report_summary = []

    # General statistics for numerical and categorical features
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns

    # Save statistical summaries
    num_summary = df[num_cols].describe().T
    num_summary.to_csv(os.path.join(output_dir, "numerical_summary.csv"))
    report_summary.append("Numerical summary saved.")

    if len(cat_cols) > 0:
        cat_summary = df[cat_cols].describe().T
        cat_summary.to_csv(os.path.join(output_dir, "categorical_summary.csv"))
        report_summary.append("Categorical summary saved.")

    # Visualizations for numeric features
    for col in num_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col].dropna(), kde=True, bins=20)
        plt.title(f"Distribution of {col}")
        plt.savefig(os.path.join(output_dir, f"{col}_hist.png"))
        plt.close()
        report_summary.append(f"Histogram (and KDE) for {col} saved.")

    # Visualizations for categorical features
    for col in cat_cols:
        plt.figure(figsize=(6,4))
        df[col].value_counts().head(15).plot(kind='bar')
        plt.title(f"Top Categories in {col}")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_bar.png"))
        plt.close()
        report_summary.append(f"Bar plot for {col} saved.")

    # Pairplot for numerical features, if not too many
    if len(num_cols) > 1 and len(num_cols) <= 6:
        sns.pairplot(df[num_cols].dropna())
        plt.savefig(os.path.join(output_dir, "pairplot.png"))
        plt.close()
        report_summary.append("Pairplot of numerical features saved.")

    # Correlation heatmap
    if len(num_cols) > 1:
        plt.figure(figsize=(8,6))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()
        report_summary.append("Correlation heatmap saved.")

    # Save report summary
    with open(os.path.join(output_dir, "eda_summary.txt"), "w") as f:
        for line in report_summary:
            f.write(line + "\n")

    print(f"EDA report generated in '{output_dir}'.")


