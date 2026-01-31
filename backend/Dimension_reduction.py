import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 2. Load Dataset
file_path = "AI-Impacts-on-Jobs-Feature-Engineered.xlsx"
df = pd.read_excel(file_path)

print("✅ Dataset Loaded")
print("Dataset Shape:", df.shape)
print(df.head())

# 3. Separate Features and Target
target_column = None  # change to your target column if exists
if target_column and target_column in df.columns:
    X = df.drop(columns=[target_column])
    y = df[target_column]
else:
    X = df.copy()
    y = None

# 4. Handle Non-Numeric Columns
X_numeric = X.select_dtypes(include=['int64', 'float64'])
print("\n✅ Numeric features shape:", X_numeric.shape)

# 5. Inspect for NaN, Inf, and Large Values
print("\n🔎 Checking for NaN, Infinite, and Large Values...\n")

# Check for NaNs
nan_counts = X_numeric.isnull().sum()
if nan_counts.sum() > 0:
    print("⚠️ Columns with NaN values:\n", nan_counts[nan_counts > 0])
else:
    print("✅ No NaN values found.")

# Replace NaN with column mean
X_numeric = X_numeric.fillna(X_numeric.mean())

# Check for Infinite values
for col in X_numeric.columns:
    X_numeric[col] = X_numeric[col].replace([np.inf, -np.inf], np.nan)


if np.isinf(X_numeric.values).any():
    print("⚠️ Infinite values detected and replaced with NaN.")
else:
    print("✅ No Infinite values found.")

# Replace any NaNs caused by Inf with column mean
X_numeric = X_numeric.fillna(X_numeric.mean())

# Check for extremely large values
max_values = X_numeric.max()
print("\n🔎 Maximum values in each numeric column:\n", max_values)

# 6. Scale the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

print("\n✅ Scaling complete. Shape after scaling:", X_scaled.shape)

# 7. Apply PCA (keep 95% variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print("\n✅ PCA Applied")
print("Original shape:", X_scaled.shape)
print("Reduced shape:", X_pca.shape)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Explained:", sum(pca.explained_variance_ratio_))

# 8. Create Reduced DataFrame
pc_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
df_reduced = pd.DataFrame(X_pca, columns=pc_columns)

if y is not None:
    df_reduced['Target'] = y

print("\n✅ First few rows of reduced dataset:\n", df_reduced.head())

# 9. Visualize First 2 Principal Components
if X_pca.shape[1] >= 2:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df_reduced['PC1'], y=df_reduced['PC2'])
    plt.title("PCA Projection (First 2 Components)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()
else:
    print("\n⚠️ Less than 2 components available for visualization.")

# 10. Save Reduced Dataset
df_reduced.to_excel("Reduced_Dataset.xlsx", index=False)
print("\n💾 Reduced dataset saved as 'Reduced_Dataset.xlsx'")
