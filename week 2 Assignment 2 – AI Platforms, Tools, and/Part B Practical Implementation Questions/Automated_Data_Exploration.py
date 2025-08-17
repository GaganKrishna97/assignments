import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data.csv")

# Summary statistics
print(df.describe())

# Correlation matrix
corr = df.corr()
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()

# Pairplot for exploratory visualization
sns.pairplot(df)
plt.show()
