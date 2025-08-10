import os
import logging
import json
from flask import Flask, request, jsonify
import mlflow.pyfunc

# Set up logging to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)

# Load the registered MLflow model
MODEL_NAME = "IrisModel"
try:
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/latest")
    logging.info(f"Successfully loaded model '{MODEL_NAME}' version 'latest'.")
except Exception as e:
    logging.error(f"Failed to load model '{MODEL_NAME}': {e}")
    model = None

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions with the Iris model."""
    if not model:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    try:
        # Get data from the POST request
        data = request.get_json(force=True)
        
        # Log the incoming request
        logging.info(f"Incoming request: {json.dumps(data)}")

        # Make prediction
        prediction = model.predict(data)[0]

        # Log the prediction result
        logging.info(f"Prediction result: {prediction}")

        # Return the prediction result
        return jsonify({"prediction": prediction})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400

# Expose a simple health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)