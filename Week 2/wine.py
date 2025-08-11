# ------------------------------------------------------------
# SECTION B - Q1: CSV Data Exploration & Visualization
# Using Wine-dataset.csv provided
# ------------------------------------------------------------

# 1. Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# 2. Load the CSV file into a Pandas DataFrame
# Make sure 'Wine-dataset.csv' is in the same folder as this script
df = pd.read_csv("Wine.csv")



# 3. Display first 10 rows to understand the structure of the dataset
print("----- FIRST 10 ROWS -----")
print(df.head(10))

# 4. Basic dataset information
print("\n----- BASIC INFO -----")
print(df.info())  # Column names, non-null counts, data types
print("\nShape of dataset (rows, columns):", df.shape)


# 5. Descriptive statistics for numerical columns
print("\n----- SUMMARY STATISTICS -----")
print(df.describe())

# 6. Check for missing values
print("\n----- MISSING VALUES PER COLUMN -----")
print(df.isnull().sum())

# 7. Handle missing values (if any)
# Here we fill missing numeric values with the column mean
df = df.fillna(df.mean(numeric_only=True))


# 8. Filter: Select wines with Alcohol content > 14
filtered_df = df[df['Alcohol'] > 14]

# 9. Sort: Sort filtered wines by Alcohol in descending order
sorted_df = filtered_df.sort_values(by='Alcohol', ascending=False)
print("\n----- FILTERED & SORTED DATA (Alcohol > 14) -----")
print(sorted_df.head())


# 10. Group By: Calculate average 'Magnesium' level by wine class
group_mean_magnesium = df.groupby('class')['Magnesium'].mean()
print("\n----- AVERAGE MAGNESIUM BY WINE CLASS -----")
print(group_mean_magnesium)

# 11. Visualization 1: Histogram of Alcohol content
plt.figure(figsize=(6,4))
plt.hist(df['Alcohol'], bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of Alcohol Content')
plt.xlabel('Alcohol (%)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 12. Visualization 2: Bar chart of average Magnesium by wine class
group_mean_magnesium.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Average Magnesium by Wine Class')
plt.xlabel('Wine Class')
plt.ylabel('Average Magnesium')
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.show()

# 13. Save the cleaned dataset to a new CSV file
df.to_csv('Wine-dataset-cleaned.csv', index=False)
print("\nCleaned dataset saved as 'WineCleaned.csv'")