import pandas as pd

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    
    if all(x in df.columns for x in ['ai_impact', 'ai_models']):
        df['impact_per_model'] = df['ai_impact'] / df['ai_models']
        df['impact_per_model'] = df['impact_per_model'].fillna(0)


    if 'ai_workload_ratio' in df.columns:
        bins = [0, 0.1, 0.2, 1]
        labels = ['Low', 'Medium', 'High']
        df['workload_category'] = pd.cut(df['ai_workload_ratio'], bins=bins, labels=labels)

    if 'domain' in df.columns:
        domain_dummies = pd.get_dummies(df['domain'], prefix='domain')
        df = pd.concat([df, domain_dummies], axis=1)

    if 'domain' in df.columns and 'ai_impact' in df.columns:
        mean_domain_impact = df.groupby('domain')['ai_impact'].transform('mean')
        df['mean_ai_impact_by_domain'] = mean_domain_impact

    if 'job_titles' in df.columns:
        df['job_title_length'] = df['job_titles'].str.len()
        df['job_title_word_count'] = df['job_titles'].str.split().apply(len)

    return df

if __name__ == "__main__":
    # Load cleaned dataset produced by main script
    cleaned_file = 'AI-Impacts-on-Jobs-Cleaned-Standardized.xlsx'
    df_cleaned = pd.read_excel(cleaned_file)

    # Perform feature engineering
    df_fe = feature_engineering(df_cleaned)

    # Save the feature engineered dataset
    output_file = 'AI-Impacts-on-Jobs-Feature-Engineered.xlsx'
    df_fe.to_excel(output_file, index=False)

    print(f"Feature engineering complete. Output saved to {output_file}")




# Load the feature engineered dataset
file_path = 'AI-Impacts-on-Jobs-Feature-Engineered.xlsx'
df_f = pd.read_excel(file_path)

# Show dataset overview
print("DATASET INFO:")
print(df_f.info())

print("\nDESCRIPTIVE STATISTICS:")
print(df_f.describe())

print("\nFIRST 10 ROWS:")
print(df_f.head(10))

print("\nMISSING VALUES PER COLUMN:")
print(df_f.isnull().sum())

# Preview some of the new engineered columns
print("\nSAMPLE OF FEATURE-ENGINEERED COLUMNS:")
cols_to_preview = ['impact_per_model', 'workload_category', 'mean_ai_impact_by_domain',
                   'job_title_length', 'job_title_word_count']

for col in cols_to_preview:
    if col in df_fe.columns:
        print(f"\nColumn: {col}")
        print(df_f[[col]].head(10))
