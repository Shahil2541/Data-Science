import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_excel('AI Impacts on jobs.xlsx', sheet_name='My_Data')

# Fill missing numeric values with median
numeric_cols = ['Tasks', 'AI models', 'AI Impact', 'AI_Workload_Ratio']
for col in numeric_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# Standardize job titles: strip, title case, expand abbreviations
def standardize_job_title(title):
    title = str(title).strip().title()
    title = re.sub(r'\bS/W Eng\b', 'Software Engineer', title)
    title = re.sub(r'\bCeo\b', 'Chief Executive Officer', title)
    # Add more abbreviation expansions as needed
    return title

df['Job titiles'] = df['Job titiles'].apply(standardize_job_title)

# Rename column for consistency
df.rename(columns={'Job titiles': 'Job titles'}, inplace=True)

# Handle infinite values in AI_Workload_Ratio
df['AI_Workload_Ratio'] = df['AI_Workload_Ratio'].replace([np.inf, -np.inf], np.nan)


df.dropna(subset=['AI_Workload_Ratio'], inplace=True)

# Remove rows where AI models == 0 to avoid division issues
df = df[df['AI models'] != 0]

# Remove outliers using IQR method
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

for col in ['AI Impact', 'Tasks', 'AI models', 'AI_Workload_Ratio']:
    df = remove_outliers_iqr(df, col)

# Scale numeric features to 0-1 range using MinMaxScaler
scaler = MinMaxScaler()
cols_to_scale = ['Tasks', 'AI models', 'AI_Workload_Ratio']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Save cleaned data to a new Excel file
df.to_excel('AI-Impacts-on-Jobs-Cleaned.xlsx', index=False)

cleaned_df = pd.read_excel('AI-Impacts-on-Jobs-Cleaned.xlsx')

# --- Validation of Cleaned Data ---
print('CLEANED DATASET INFO:')
print(cleaned_df.info())

print('\nDESCRIPTIVE STATISTICS:')
print(cleaned_df.describe())

print('\nFIRST 5 ROWS:')
print(cleaned_df.head())

print('\nMISSING VALUES PER COLUMN:')
print(cleaned_df.isnull().sum())

# --- Outlier Detection: Find which rows were removed ---
# We'll use a merge with indicator to find dropped rows
comparison_cols = [col for col in df.columns if col in cleaned_df.columns]
outliers_removed = df.merge(cleaned_df, how='outer', indicator=True, on=comparison_cols)
outliers_only = outliers_removed[outliers_removed['_merge'] == 'left_only']

print(f"\nNumber of outlier rows removed: {len(outliers_only)}")
if not outliers_only.empty:
    print('\nSample of outlier rows removed:')
    print(outliers_only[comparison_cols].head())
else:
    print('No outliers were removed.')









# # Load the cleaned dataset to validate
# cleaned_df = pd.read_excel('AI-Impacts-on-Jobs-Cleaned.xlsx')

# # Print dataset info
# print(cleaned_df.info())

# # Print descriptive statistics for numeric columns
# print(cleaned_df.describe())

# # Display the first 5 rows
# print(cleaned_df.head())

# # Check for any missing values in each column
# print("Missing values per column:")
# print(cleaned_df.isnull().sum())




