Decision Tree Prediction API

This project provides a RESTful API using Flask to serve predictions from a trained Decision Tree model. It supports preprocessing (including scaling and encoding), and it handles categorical and numerical features to predict a service likely to be purchased next.
🧠 Model Overview

    Model Type: Decision Tree Classifier

    Features: Both numerical and categorical

    Target: Second_Purchase_Service

    Input Format: JSON

    Output: Predicted label (e.g., Auto, Health, Real Estate)

📁 Project Structure

ML_PREDICTION_SERVICE/
│
├── decision_tree_model.pkl                 # Trained Decision Tree model
├── scaler.joblib                           # StandardScaler for numerical features
├── encoder_<feature>.joblib                # LabelEncoders for categorical features
├── encoder_Second_Purchase_Service.joblib  # Target encoder (optional)
├── app.py                                  # Flask API script

▶️ Running the API
1. Clone the project

git clone <your-repo-url>
cd ML_PREDICTION_SERVICE

2. Install dependencies

pip install -r requirements.txt

Example requirements.txt:

Flask
pandas
numpy
scikit-learn
joblib

3. Start the Flask server

python app.py

The API will run on: http://0.0.0.0:5000
🧪 API Endpoints
GET /

Returns a welcome message.

Request:

curl http://localhost:5000/

Response:

{ "message": "Welcome to the Decision Tree Prediction API. Use POST /predict to get predictions." }

POST /predict

Predicts the next purchase service based on input features.

Headers:

Content-Type: application/json

Example Request Body:

{
  "age": 35,
  "gender": "Male",
  "region": "Tunis",
  "maritalStatus": "Married",
  "occupation": "Engineer",
  "incomeLevel": 45000,
  "tenureYears": 5,
  "satisfactionScore": 3,
  "interestedInOtherProducts": "Yes",
  "preferredCommunicationChannel": "SMS",
  "purchaseAmount": 3000,
  "isBookmarked": 1,
  "currentProduct": "Auto",
  "bookmarkedService": "Real Estate",
  "isHealthViewed": 1,
  "isAutoViewed": 0,
  "isRealEstateViewed": 1
}

Success Response:

{
  "prediction": "Health",
  "prediction_value": 1
}

⚙️ Preprocessing Details

    Numerical Scaling: Done via StandardScaler

    Categorical Encoding: LabelEncoders saved as .joblib files

    Fallbacks: If encoders are missing, warnings are logged

    Missing Fields: Non-required fields (e.g., marketingInteraction) are allowed to be null

📌 Notes

    Make sure model and encoder files are placed in the same directory as specified in MODEL_DIR.

    The server logs detailed processing steps and potential issues to the console.

✅ Status

✅ Fully functional
✅ Handles missing encoders gracefully
✅ Converts NumPy types for JSON compatibility
✅ Scales numeric fields and encodes categories
