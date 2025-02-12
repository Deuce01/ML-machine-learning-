Data Preprocessing Documentation

Introduction

In this document, I outline the steps I followed to preprocess a dataset using Python and various data science libraries, including Pandas, NumPy, Seaborn, and Scikit-Learn. The goal of this preprocessing was to clean, transform, and normalize the dataset for further analysis or machine learning applications.

1. Loading the Dataset

I loaded the dataset into a Pandas DataFrame using the read_csv function. The dataset file was named data.csv, and I ensured it was available in the working directory before loading it.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv('data.csv')

2. Initial Exploration

To understand the structure and content of the dataset, I performed the following exploratory steps:

Displayed dataset information using df.info()

Previewed the first five rows using df.head()

Generated summary statistics using df.describe(include='all')

# Display dataset information
print("Dataset Information:")
df.info()

# Show first few rows
print("\nFirst 5 rows:")
print(df.head())

# Summary statistics
print("\nSummary statistics:")
print(df.describe(include='all'))

3. Handling Missing Values

I checked for missing values and visualized them using a heatmap.

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Visualizing missing values
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

To handle missing values:

I filled numerical columns with their median values.

I filled categorical columns with their mode.

# Handling missing values
df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numeric columns with median
df.fillna(df.mode().iloc[0], inplace=True)  # Fill categorical columns with mode

4. Encoding Categorical Data

Since the dataset contained categorical variables, I used LabelEncoder to convert them into numerical values.

# Encoding categorical data
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

5. Feature Normalization

To ensure that all features had the same scale, I used StandardScaler for normalization.

# Normalizing features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

6. Saving the Processed Data

Finally, I saved the processed dataset to a new CSV file named processed_data.csv.

# Save processed data
df_scaled.to_csv('processed_data.csv', index=False)

Conclusion

Through this preprocessing workflow, I successfully cleaned the dataset, handled missing values, encoded categorical variables, and normalized the data for further analysis. This processed dataset is now ready for machine learning models or other analytical tasks.

