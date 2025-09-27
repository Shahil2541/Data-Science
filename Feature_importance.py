import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_excel("Reduced_Dataset_RF.xlsx")
X = df.drop(columns=['ai_impact'])
y = df['ai_impact']

rf = RandomForestRegressor()
rf.fit(X, y)

# Feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values(by='importance', ascending=False)

print(importance)


# Partial Dependence  for 'impact_per_model'

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.inspection import partial_dependence, PartialDependenceDisplay

# # Load your dataset
# df = pd.read_excel("Reduced_Dataset_RF.xlsx")

# # Separate features and target
# X = df.drop(columns=['ai_impact'])
# y = df['ai_impact']

# # Train Random Forest
# rf = RandomForestRegressor()
# rf.fit(X, y)

# # Partial Dependence Plot for 'impact_per_model'
# fig, ax = plt.subplots(figsize=(8,5))
# PartialDependenceDisplay.from_estimator(rf, X, ['impact_per_model'], ax=ax)
# plt.title('Partial Dependence of ai_impacts on impact_per_model')
# plt.show()

