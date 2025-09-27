import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder

# 2. Load Dataset
file_path = "AI-Impacts-on-Jobs-Feature-Engineered.xlsx"
df = pd.read_excel(file_path)

print("✅ Dataset Loaded")
print("Dataset Shape:", df.shape)
print(df.head())

# 3. Define Target Column
target_column = "ai_impact"  

if target_column not in df.columns:
    raise ValueError(f"❌ Target column '{target_column}' not found in dataset!")

# Separate features (X) and target (y)
X = df.drop(columns=[target_column])
y = df[target_column]

# 4. Encode Categorical Target (if needed)
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)
    print("✅ Target column encoded.")

# 5. Keep Only Numeric Features
X_numeric = X.select_dtypes(include=['int64', 'float64'])

# Handle NaN/inf values
X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
X_numeric = X_numeric.fillna(X_numeric.mean())

print("✅ Numeric Features Shape:", X_numeric.shape)

# 6. Train RandomForest for Feature Importance
# Choose classifier or regressor based on target type
if len(np.unique(y)) <= 20:  # Assuming classification for few unique labels
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
else:
    rf = RandomForestRegressor(n_estimators=200, random_state=42)

rf.fit(X_numeric, y)

# Get feature importances
importances = rf.feature_importances_
feature_names = X_numeric.columns

# Create a DataFrame of feature importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n🔎 Feature Importances:")
print(importance_df)

# 7. Select Top Features (Dimension Reduction)
#  Keep features above a certain importance threshold
selector = SelectFromModel(rf, threshold="median", prefit=True)  # keep features above median importance
X_reduced = selector.transform(X_numeric)

selected_features = feature_names[selector.get_support()]
print("\n✅ Selected Features for Reduced Dataset:")
print(selected_features.tolist())

print(f"\nOriginal shape: {X_numeric.shape}")
print(f"Reduced shape: {X_reduced.shape}")

# 8. Save Reduced Dataset
df_reduced = pd.DataFrame(X_reduced, columns=selected_features)

# Add target back
df_reduced[target_column] = y

df_reduced.to_excel("Reduced_Dataset_RF.xlsx", index=False)
print("\n💾 Reduced dataset saved as 'Reduced_Dataset_RF.xlsx'")
