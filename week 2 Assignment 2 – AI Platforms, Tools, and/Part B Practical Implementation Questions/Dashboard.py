import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
df.columns = df.columns.str.strip()  # <-- This line fixes the space issue

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].hist(df["Alcohol"])
axs[0, 0].set_title("Alcohol Distribution")
axs[0, 1].scatter(df["Alcohol"], df["Malic acid"])
axs[0, 1].set_title("Alcohol vs Malic acid")
axs[1, 0].boxplot(df["Hue"])
axs[1, 0].set_title("Hue Boxplot")
axs[1, 1].plot(df["Proline"])
axs[1, 1].set_title("Proline Trend")
plt.tight_layout()
plt.show()
