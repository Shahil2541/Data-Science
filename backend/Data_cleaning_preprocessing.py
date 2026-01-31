import pandas as pd
import numpy as np
import re
import string
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Load original data
file_path = 'AI Impacts on Jobs.xlsx'
original_df = pd.read_excel(file_path, sheet_name='My_Data')

# Clean column names: lowercase, replace spaces with underscores, strip spaces
original_df.columns = original_df.columns.str.strip().str.lower().str.replace(' ', '_')

# Fix common typos in column names (e.g. 'job_titiles' -> 'job_titles')
if 'job_titiles' in original_df.columns:
    original_df.rename(columns={'job_titiles': 'job_titles'}, inplace=True)

print('Column names after cleaning:', original_df.columns.tolist())

# Copy to cleaned_df for processing
df = original_df.copy()

# Fill missing numeric values with median
numeric_cols = ['tasks', 'ai_models', 'ai_impact', 'ai_workload_ratio']
for col in numeric_cols:
    if col in df.columns and df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f'Filled missing values in {col} with median: {median_val}')

# Advanced job title standardization function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def standardize_job_title(title):
    title = str(title).strip().lower()
    # Expand abbreviations
    title = re.sub(r'\bs/w eng\b', 'software engineer', title)
    title = re.sub(r'\bceo\b', 'chief executive officer', title)
    title = re.sub(r'\bcto\b', 'chief technology officer', title)
    title = re.sub(r'\bhr\b', 'human resources', title)
    title = re.sub(r'\bqa\b', 'quality assurance', title)
    # Remove punctuation
    title = title.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(title)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Remove non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]
    return ' '.join(tokens)

# Apply standardization to job_titles column, if present
if 'job_titles' in df.columns:
    df['job_titles'] = df['job_titles'].apply(standardize_job_title)
else:
    print('Job title column not found!')

# Handle infinite values in ai_workload_ratio column
if 'ai_workload_ratio' in df.columns:
    inf_count = df['ai_workload_ratio'].isin([np.inf, -np.inf]).sum()
    if inf_count > 0:
        print(f'Replacing {inf_count} infinite values in ai_workload_ratio with NaN')
        df['ai_workload_ratio'] = df['ai_workload_ratio'].replace([np.inf, -np.inf], np.nan)

    df.dropna(subset=['ai_workload_ratio'], inplace=True)

# Remove rows where ai_models == 0 to avoid division issues
if 'ai_models' in df.columns:
    zero_ai_models_count = (df['ai_models'] == 0).sum()
    if zero_ai_models_count > 0:
        print(f'Removing {zero_ai_models_count} rows with ai_models == 0')
        df = df[df['ai_models'] != 0]

# Removing outliers using IQR for specified columns
def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
    print(f'Removed {len(dataframe) - len(filtered_df)} outliers from {column}')
    return filtered_df

print(f"Original dataset shape before outlier removal: {df.shape}")

for col in ['ai_impact', 'tasks', 'ai_models', 'ai_workload_ratio']:
    if col in df.columns:
        df = remove_outliers_iqr(df, col)

print(f"Dataset shape after outlier removal: {df.shape}")
num_outliers_removed = len(original_df) - len(df)
print(f"Total outliers removed: {num_outliers_removed}")

# Scale numeric features to 0-1 range
scaler = MinMaxScaler()
cols_to_scale = [col for col in ['tasks', 'ai_models', 'ai_workload_ratio'] if col in df.columns]
if cols_to_scale:
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    print(f'Scaled columns: {cols_to_scale}')

print(f"Final cleaned dataset shape: {df.shape}")

# Save cleaned data to new Excel file
output_path = 'AI-Impacts-on-Jobs-Cleaned-Standardized.xlsx'
df.to_excel(output_path, index=False)
print(f"Data cleaning and preprocessing complete. Output saved to {output_path}")

# Validation of cleaned data by loading saved file
cleaned_df = pd.read_excel(output_path)
print('\nCLEANED DATASET INFO:')
print(cleaned_df.info())
print('\nDESCRIPTIVE STATISTICS:')
print(cleaned_df.describe())
print('\nFIRST 5 ROWS:')
print(cleaned_df.head())
print('\nMISSING VALUES PER COLUMN:')
print(cleaned_df.isnull().sum())

