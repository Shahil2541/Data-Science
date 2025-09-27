import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# 2. Load reduced dataset
df = pd.read_excel("Reduced_Dataset_RF.xlsx")

# 3. Define features (X) and target (y)
target_column = 'ai_impact'        # <-- target column
X = df.drop(columns=[target_column])
y = df[target_column]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# 6. Predictions
y_pred = rf_model.predict(X_test)

# 7. Evaluation Metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
pearson_corr, _ = pearsonr(y_test, y_pred)

print("✅ Random Forest Regressor Results:")
print(f"R² Score             : {r2:.3f}")
print(f"RMSE                 : {rmse:.3f}")
print(f"Pearson Correlation  : {pearson_corr:.3f}")

# 8. Feature Importance Plot
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=True).plot(kind='barh', figsize=(8,6))
plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
