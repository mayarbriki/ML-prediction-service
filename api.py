from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialiser Flask
app = Flask(__name__)

# Charger le modèle
model = joblib.load("xgboost_fraud_model_tuned.pkl")

# Route de prédiction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = [[
        data['Product_Type'],
        data['Policy_Amount'],
        data['Claim_Amount'],
        data['Claim_Type'],
        data['Previous_Claims_History'],
        data['Driving_Record'],
        data['Credit_Score'],
        data['Risk_Profile'],
        data['Claim_to_Policy_Ratio'],
        data['Has_Claim_History'],
        data['Days_Since_Claim']
    ]]

    prediction = model.predict(features)[0]
    label = "Fraud" if prediction == 1 else "Not Fraud"

    return jsonify({"prediction": int(prediction), "label": label})

if __name__ == '__main__':
    app.run(debug=True)