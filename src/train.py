import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn

# Set the MLflow tracking URI to a local directory
mlflow.set_tracking_uri("file:///../mlruns")
os.makedirs("mlruns", exist_ok=True)


def train_and_track_models():
    """Trains two models, tracks them with MLflow, and registers the best one."""

    # Load the preprocessed data
    df = pd.read_csv('./../data/iris.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dictionary to hold the trained models and their metrics
    models = {
        'LogisticRegression': LogisticRegression(max_iter=200),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100)
    }

    best_model = None
    best_accuracy = 0
    best_run_id = None

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Log parameters
            mlflow.log_params(model.get_params())

            # Log metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)

            # Log the model
            mlflow.sklearn.log_model(model, "iris-model")
            
            print(f"Logged {model_name} with accuracy: {accuracy}")
            
            # Check for the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_run_id = mlflow.active_run().info.run_id

    # Register the best model
    if best_model:
        # Get the run ID of the best model
        client = mlflow.tracking.MlflowClient()
        source_uri = f"runs:/{best_run_id}/iris-model"
        mlflow.register_model(model_uri=source_uri, name="IrisModel")
        print(f"\nBest model (with accuracy {best_accuracy}) registered as 'IrisModel'.")

if __name__ == '__main__':
    train_and_track_models()
