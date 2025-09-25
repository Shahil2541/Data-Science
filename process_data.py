import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


file_path = "Dataset.csv"
df = pd.read_csv('Dataset.csv', quotechar='"', skipinitialspace=True, on_bad_lines='skip')

print("--- First 5 Rows of the Dataset ---")
print(df.head())

print("\n--- Structure and Data Types of the Dataset ---")
# .info() is similar to R's str()
df.info()

print("\n--- Statistical Summary of the Dataset ---")
# .describe() is similar to R's summary() for numerical data
print(df.describe())


# -------------------------------------------------------------------
# 3. DATA CLEANING AND TRANSFORMATION
# -------------------------------------------------------------------

# Convert text "N/A" into NumPy's proper NaN representation (np.nan)
df.replace("N/A", np.nan, inplace=True)
# The `inplace=True` argument modifies the DataFrame directly.

# Impute missing values
# In pandas, we use the .fillna() method
df['TechAdoption_pct'].fillna(0, inplace=True)
df['RelevantTechnology'].fillna("None", inplace=True)

print("\n--- Count of Missing Values After Imputation ---")
print(df.isnull().sum())

# Convert object columns to 'category' type, which is similar to R's factors
cols_to_factor = ['JobCategory', 'ReskillingPriority', 'RequiredEducationLevel',
                  'CognitiveSkill_Importance', 'SocialSkill_Importance',
                  'JobLocationFlexibility', 'AI_Tool_Maturity', 'DataAvailability_Impact']

for col in cols_to_factor:
    df[col] = df[col].astype('category')

print("\n--- Data Types After Converting to Categories ---")
df.info()


# -------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# -------------------------------------------------------------------

# In pandas, we create new columns by direct assignment
df['SalaryChange_USD'] = df['ProjectedSalary_USD_PostAI'] - df['AvgSalary_USD_PreAI']
df['SalaryChange_pct'] = round((df['SalaryChange_USD'] / df['AvgSalary_USD_PreAI']) * 100, 2)

# Create a binary feature for job growth (1 if 'Growing', 0 otherwise)
df['IsGrowing'] = (df['JobCategory'] == 'Growing').astype(int)

# Create a ratio of Automation to Augmentation
df['Automation_Augmentation_Ratio'] = df['TaskAutomation_pct'] / (df['TaskAugmentation_pct'] + 1)


print("\n--- Head of Preprocessed Dataset (Before Dimension Reduction) ---")
# Use .iloc to select columns by position, similar to R's df[, 1:10]
print(df.iloc[:, :10].head())


# -------------------------------------------------------------------
# 5. DIMENSION REDUCTION USING PCA
# -------------------------------------------------------------------

# --- Step 5.1: Select only numerical data for PCA ---
# The select_dtypes method is perfect for this.
numerical_df = df.select_dtypes(include=np.number)

print("\n--- Columns selected for PCA ---")
print(numerical_df.columns.tolist())

# --- Step 5.2: Scale the data ---
# We use StandardScaler from scikit-learn, which is standard practice.
scaler = StandardScaler()
scaled_numerical_df = scaler.fit_transform(numerical_df)

# --- Step 5.3: Perform PCA ---
# We use the PCA class from scikit-learn.
pca = PCA()
pca_result = pca.fit_transform(scaled_numerical_df)

# --- Step 5.4: Analyze PCA Results ---
# The 'explained_variance_ratio_' attribute shows the variance explained by each component.
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\n--- Summary of PCA Results ---")
print(f"Explained Variance per PC: {np.round(explained_variance, 3)}")
print(f"Cumulative Variance:       {np.round(cumulative_variance, 3)}")

# --- Step 5.5: Visualize the PCA Results (Scree Plot) ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.grid(True)
print("\n--- Generating Scree Plot... ---")
plt.show()

# --- Step 5.6: Create a new, reduced-dimension dataset ---
# Let's choose the first 4 PCs.
num_components_to_keep = 4
df_reduced = pd.DataFrame(data=pca_result[:, :num_components_to_keep],
                          columns=[f'PC{i+1}' for i in range(num_components_to_keep)])

print(f"\n--- Head of the New Dataset with {num_components_to_keep} Dimensions (Principal Components) ---")
print(df_reduced.head())

# --- Step 5.7: Combine PCs with key original labels for context ---
# Use pd.concat to join the new PC dataframe with key categorical columns.
df_final = pd.concat([df_reduced, df[['JobID', 'JobTitle', 'JobCategory', 'IsGrowing']]], axis=1)

print("\n--- Head of Final Dataset (PCs + Original Labels) ---")
print(df_final.head())

# --- Step 5.8: Visualize the final PCA result ---
# Use seaborn for an attractive and informative scatter plot.
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_final, x='PC1', y='PC2', hue='JobCategory', palette='viridis', s=100, alpha=0.8)
plt.title('Jobs Visualized by First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Job Category')
plt.grid(True)
print("\n--- Generating PCA Visualization Plot... ---")
plt.show()

print("\n--- Full Preprocessing and Dimension Reduction Complete ---")