from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Define paths for model files
MODEL_DIR = r"C:\Users\Mayar\Downloads\ML_PREDICTION_SERVICE"  # Update this to where your files are located
MODEL_PATH = os.path.join(MODEL_DIR, "decision_tree_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

# Initialize label_encoders dictionary
label_encoders = {}

# Load the trained Decision Tree model
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully", flush=True)
except FileNotFoundError:
    raise Exception(f"Model file not found at: {MODEL_PATH}")

# Load the target label encoder for Second_Purchase_Service
TARGET_ENCODER_PATH = os.path.join(MODEL_DIR, "encoder_Second_Purchase_Service.joblib")
if os.path.exists(TARGET_ENCODER_PATH):
    label_encoders['Second_Purchase_Service'] = joblib.load(TARGET_ENCODER_PATH)
    print("Loaded target encoder for Second_Purchase_Service", flush=True)
else:
    print(f"Warning: Target encoder for Second_Purchase_Service not found at {TARGET_ENCODER_PATH}", flush=True)

# Load or create preprocessing objects
try:
    # Try to load the scaler
    try:
        scaler = joblib.load(SCALER_PATH)
        # Test if the scaler is fitted
        test_data = np.array([[1, 2, 3, 4, 5, 6]])
        scaler.transform(test_data)
        print("Scaler loaded and validated successfully", flush=True)
    except (FileNotFoundError, ValueError, AttributeError) as e:
        print(f"Scaler issue: {str(e)}", flush=True)
        print("Creating a new scaler...", flush=True)

        # Create a new scaler with sample data
        sample_data = pd.DataFrame({
            'Age': np.random.randint(18, 65, 100),
            'Income_Level': np.random.randint(20000, 100000, 100),
            'Tenure_Years': np.random.randint(1, 20, 100),
            'Satisfaction_Score': np.random.randint(1, 5, 100),
            'Purchase_Amount': np.random.randint(1000, 10000, 100),
            'Is_Bookmarked': np.random.randint(0, 2, 100)
        })

        scaler = StandardScaler()
        scaler.fit(sample_data)

        # Save the new scaler
        joblib.dump(scaler, SCALER_PATH)
        print(f"New scaler created and saved to {SCALER_PATH}", flush=True)

    # Define categorical columns and their possible values
    categorical_cols_info = {
        'Gender': ['Female', 'Male'],
        'Region': ['Ariana', 'Beja', 'Ben Arous', 'Bizerte', 'Gabes', 'Gafsa', 'Jendouba', 'Kairouan', 'Kasserine',
                   'Kebili', 'Kef', 'Mahdia', 'Manouba', 'Medenine', 'Monastir', 'Nabeul', 'Sfax', 'Sidi Bouzid',
                   'Siliana', 'Sousse', 'Tataouine', 'Tozeur', 'Tunis', 'Zaghouan'],
        'Marital_Status': ['Divorced', 'Married', 'Single', 'Widowed'],
        'Occupation': ['Administrative', 'Engineer', 'Executive', 'Other', 'Sales', 'Self-employed', 'Student',
                       'Technician'],
        'Interested_in_Other_Products': ['No', 'Yes'],
        'Marketing_Interaction': ['Email', 'None', 'Phone', 'SMS', 'Social Media'],
        'Preferred Communication Channel': ['Email', 'Phone', 'SMS'],
        'Current_Product': ['Auto', 'Health', 'None', 'Real Estate'],
        'Bookmarked_Service': ['Auto', 'Health', 'None', 'Real Estate'],
        'Is_Health_Viewed': [0, 1],
        'Is_Auto_Viewed': [0, 1],
        'Is_RealEstate_Viewed': [0, 1]
    }

    # Load label encoders
    for col in categorical_cols_info.keys():
        encoder_path = os.path.join(MODEL_DIR, f"encoder_{col}.joblib")
        if os.path.exists(encoder_path):
            label_encoders[col] = joblib.load(encoder_path)
            print(f"Loaded encoder for {col}", flush=True)
        else:
            print(f"Warning: Encoder for {col} not found at {encoder_path}", flush=True)

except Exception as e:
    print(f"Error during initialization: {str(e)}", flush=True)
    raise

# Numerical features (these will be scaled)
numerical_cols = ['Age', 'Income_Level', 'Tenure_Years', 'Satisfaction_Score', 'Purchase_Amount', 'Is_Bookmarked']

# Map API input fields to DataFrame columns
FIELD_MAPPING = {
    'age': 'Age',
    'gender': 'Gender',
    'region': 'Region',
    'maritalStatus': 'Marital_Status',
    'occupation': 'Occupation',
    'incomeLevel': 'Income_Level',
    'tenureYears': 'Tenure_Years',
    'satisfactionScore': 'Satisfaction_Score',
    'interestedInOtherProducts': 'Interested_in_Other_Products',
    'marketingInteraction': 'Marketing_Interaction',
    'preferredCommunicationChannel': 'Preferred Communication Channel',
    'purchaseAmount': 'Purchase_Amount',
    'isBookmarked': 'Is_Bookmarked',
    'currentProduct': 'Current_Product',
    'bookmarkedService': 'Bookmarked_Service',
    'isHealthViewed': 'Is_Health_Viewed',
    'isAutoViewed': 'Is_Auto_Viewed',
    'isRealEstateViewed': 'Is_RealEstate_Viewed'
}


# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# Configure Flask to use our custom JSON encoder
app.json_encoder = NumpyEncoder


# Helper function to ensure all NumPy types are converted to Python types
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj


@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the Decision Tree Prediction API. Use POST /predict to get predictions.'})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure the Content-Type is application/json
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 415

        # Get JSON data from request
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Log the input data for debugging
        print(f"Received input: {data}", flush=True)

        # Validate required fields
        required_fields = list(FIELD_MAPPING.keys())
        if 'marketingInteraction' in required_fields:
            required_fields.remove('marketingInteraction')

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {missing_fields}'}), 400

        # Create DataFrame with properly mapped column names
        input_data = {}
        for api_field, df_column in FIELD_MAPPING.items():
            if api_field in data:
                input_data[df_column] = [data[api_field]]
            elif df_column == 'Marketing_Interaction':
                input_data[df_column] = [np.nan]
            else:
                input_data[df_column] = [0]

        input_df = pd.DataFrame(input_data)
        print(f"Processed input dataframe:\n{input_df}", flush=True)

        # Create a copy for processing
        processed_df = input_df.copy()

        # Normalize binary fields to 0 or 1
        binary_fields = ['Is_Health_Viewed', 'Is_Auto_Viewed', 'Is_RealEstate_Viewed']
        for field in binary_fields:
            processed_df[field] = processed_df[field].apply(lambda x: 1 if x > 0 else 0)

        # Scale numerical features
        numerical_data = processed_df[numerical_cols].copy()
        print(f"Numerical data before scaling:\n{numerical_data}", flush=True)

        scaled_data = scaler.transform(numerical_data)
        print(f"Scaled numerical data:\n{scaled_data}", flush=True)

        processed_df[numerical_cols] = scaled_data

        # One-hot encode categorical features
        processed_dummies = pd.get_dummies(processed_df)
        print(f"One-hot encoded DataFrame (initial):\n{processed_dummies.head()}", flush=True)

        # Get the feature names from the model
        model_feature_names = model.feature_names_in_
        print(f"Model expects these features: {model_feature_names}", flush=True)

        # Create a DataFrame with all expected model features, filled with zeros
        final_df = pd.DataFrame(0, index=[0], columns=model_feature_names)

        # Fill in the values we have from our processed data
        for col in processed_dummies.columns:
            if col in final_df.columns:
                final_df[col] = processed_dummies[col].values

        print(f"Final prepared DataFrame for prediction:\n{final_df.head()}", flush=True)

        # Make prediction
        print("Attempting prediction...", flush=True)
        prediction_encoded = model.predict(final_df)
        print(f"Raw prediction: {prediction_encoded}", flush=True)

        # Convert NumPy int64 to Python int - this is where the error was happening
        # Force conversion to Python built-in int
        prediction_value = int(prediction_encoded[0].item())
        print(f"Prediction value (converted to Python int): {prediction_value} (type: {type(prediction_value)})",
              flush=True)

        # Decode the prediction
        # Default mapping in case the encoder doesn't work
        service_mapping = {0: "Auto", 1: "Health", 2: "Real Estate"}
        prediction_label = service_mapping.get(prediction_value, "None")  # Map numeric value to service name

        if 'Second_Purchase_Service' in label_encoders:
            try:
                encoder = label_encoders['Second_Purchase_Service']
                if hasattr(encoder, 'classes_') and prediction_value in encoder.transform(encoder.classes_).tolist():
                    print(f"Label encoder classes: {encoder.classes_}", flush=True)
                    # Try to get the actual service name from the encoder
                    decoded_label = encoder.inverse_transform([prediction_value])[0]

                    # Check if the decoded label is still numeric
                    if isinstance(decoded_label, (np.integer, int)) or (
                            isinstance(decoded_label, str) and decoded_label.isdigit()):
                        # If it's still a number, use our mapping
                        numeric_value = int(decoded_label) if isinstance(decoded_label, str) else decoded_label
                        prediction_label = service_mapping.get(numeric_value, f"Service_{numeric_value}")
                    else:
                        # If it's already a string like "Auto", use it directly
                        prediction_label = str(decoded_label)

                    print(f"Decoded next purchase prediction: {prediction_label} (type: {type(prediction_label)})",
                          flush=True)
                else:
                    print(f"Prediction value {prediction_value} not found in encoder classes, using mapping",
                          flush=True)
            except ValueError as ve:
                print(f"Error decoding prediction: {str(ve)}, using mapping", flush=True)
            except Exception as e:
                print(f"Unexpected error decoding prediction: {str(e)}, using mapping", flush=True)
        else:
            print("No label encoder found for Second_Purchase_Service, using default mapping", flush=True)

        # Make sure we're absolutely converting everything to Python types
        response_data = {
            'prediction': prediction_label,
            'prediction_value': int(prediction_value)  # Force Python int
        }

        # Use our helper function to ensure all NumPy types are converted
        response_data = convert_numpy_types(response_data)

        print(
            f"Response data types: prediction={type(response_data['prediction'])}, prediction_value={type(response_data['prediction_value'])}",
            flush=True)

        return jsonify(response_data)

    except ValueError as ve:
        print(f"ValueError: {str(ve)}", flush=True)
        return jsonify({'error': f'JSON parsing error: {str(ve)}'}), 400
    except Exception as e:
        print(f"General error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)