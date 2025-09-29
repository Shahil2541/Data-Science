from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

app = Flask(__name__)

# -------------------------------
# 1. Load and train model
# -------------------------------
df = pd.read_excel("Reduced_Dataset_RF.xlsx")
target_column = 'ai_impact'
X = df.drop(columns=[target_column])
y = df[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Save model (optional)
with open("model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# Compute evaluation metrics
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
pearson_corr, _ = pearsonr(y_test, y_pred)

feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)


@app.route('/')
def home():
    return render_template("index.html",
                           r2=round(r2, 3),
                           rmse=round(rmse, 3),
                           pearson=round(pearson_corr, 3),
                           features=feature_importances.sort_values(ascending=False).to_dict())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Ensure feature order matches training
        input_df = pd.DataFrame([data])
        
        # Handle missing columns
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0  # or appropriate default value
        
        input_df = input_df[X.columns]  # reorder columns to match training data
        prediction = rf_model.predict(input_df)[0]
        
        # Ensure prediction is within reasonable bounds
        prediction = max(0, min(100, prediction))

        return jsonify({
            "prediction": round(prediction, 3),
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

if __name__ == '__main__':
    app.run(debug=True)

