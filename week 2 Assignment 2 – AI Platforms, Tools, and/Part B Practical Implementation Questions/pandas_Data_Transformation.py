import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataTransformationPipeline:
    def __init__(self, missing_strategy='mean', outlier_strategy='zscore', scaling='standard'):
        self.missing_strategy = missing_strategy
        self.outlier_strategy = outlier_strategy
        self.scaling = scaling
        self.log = []
        self.scaler = None

    def handle_missing(self, df):
        strategy = self.missing_strategy.lower()
        for col in df.select_dtypes(include=np.number).columns:
            if strategy == 'mean':
                val = df[col].mean()
                df[col].fillna(val, inplace=True)
                self.log.append(f'Filled missing in {col} with mean ({val:.3f})')
            elif strategy == 'median':
                val = df[col].median()
                df[col].fillna(val, inplace=True)
                self.log.append(f'Filled missing in {col} with median ({val:.3f})')
            elif strategy == 'mode':
                val = df[col].mode()[0]
                df[col].fillna(val, inplace=True)
                self.log.append(f'Filled missing in {col} with mode ({val})')
            elif strategy == 'drop':
                before = df.shape[0]
                df.dropna(subset=[col], inplace=True)
                after = df.shape[0]
                self.log.append(f'Dropped {before-after} rows with missing {col}')
        return df

    def remove_outliers(self, df):
        if self.outlier_strategy == 'zscore':
            for col in df.select_dtypes(include=np.number).columns:
                z = (df[col] - df[col].mean()) / df[col].std()
                outliers = (np.abs(z) > 3)
                before = df.shape[0]
                df = df[~outliers]
                after = df.shape[0]
                n_removed = before - after
                self.log.append(f'Removed {n_removed} outliers from {col} using z-score')
        elif self.outlier_strategy == 'iqr':
            for col in df.select_dtypes(include=np.number).columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outliers = (df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))
                before = df.shape[0]
                df = df[~outliers]
                after = df.shape[0]
                n_removed = before - after
                self.log.append(f'Removed {n_removed} outliers from {col} using IQR')
        return df

    def scale_features(self, df):
        numeric_cols = df.select_dtypes(include=np.number).columns
        if self.scaling == 'standard':
            self.scaler = StandardScaler()
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            self.log.append('Standardized numeric features (mean=0, std=1)')
        elif self.scaling == 'minmax':
            self.scaler = MinMaxScaler()
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            self.log.append('Normalized numeric features (range 0-1)')
        return df

    def transform(self, df):
        self.log = []  # Reset log for this run
        df = self.handle_missing(df)
        df = self.remove_outliers(df)
        df = self.scale_features(df)
        self.log.append('Transformation pipeline complete')
        return df

    def get_log(self):
        return self.log

# --- USAGE EXAMPLE ---

# Sample data
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 100],    # Intentional outlier!
    'B': [10, 20, 30, np.nan, 50],
    'C': [7, 8, 9, 10, 11]
})

pipeline = DataTransformationPipeline(
    missing_strategy='mean',
    outlier_strategy='zscore',
    scaling='standard'
)

df_transformed = pipeline.transform(df)
print("Transformed Data:\n", df_transformed)
print("\nPipeline Log:")
for entry in pipeline.get_log():
    print("-", entry)
