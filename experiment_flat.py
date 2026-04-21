import mlflow
import yaml
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# Load configuration from YAML file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set the tracking server URI and the experiment name
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("1_Flat_Comparison")

# Define run name based on hyperparameters from config
run_name = f"RF_Trees_{config['model']['n_estimators']}_Depth_{config['model']['max_depth']}"

# Start an MLflow run to log the experiment
with mlflow.start_run(run_name=run_name):
    # --- Data Acquisition with Caching ---
    local_path = config['data']['local_path']
    if os.path.exists(local_path):
        # Load dataset from local file if it exists
        print(f"Loading from cache: {local_path}")
        mnist = joblib.load(local_path)
    else:
        # Download dataset from OpenML if cache is missing
        print("Cache not found. Downloading MNIST from OpenML...")
        mnist = fetch_openml(config['data']['dataset_name'], version=1, as_frame=False, parser='auto')
        # Save downloaded data for future use
        joblib.dump(mnist, local_path)
        print(f"Data saved to {local_path}")

    # Split data into training and testing sets
    X, y = mnist.data, mnist.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    # Log parameters to MLflow
    mlflow.log_params(config['data'])
    mlflow.log_params(config['model'])

    # Initialize and train the Random Forest model
    model = RandomForestClassifier(
        n_estimators=config['model']['n_estimators'],
        max_depth=config['model']['max_depth'],
        n_jobs=config['model']['n_jobs'],
        random_state=config['data']['random_state']
    )
    model.fit(X_train, y_train)

    # Evaluate the model and log accuracy metric
    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)

    # Log the Model Artifact
    # Save the model locally and then log it to MLflow
    model_filename = f"model_trees_{config['model']['n_estimators']}_depth_{config['model']['max_depth']}.pkl"
    joblib.dump(model, model_filename)
    mlflow.log_artifact(model_filename)

    # Generate and Log Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test), ax=ax)

    # Save the plot locally and log it as an artifact in MLflow
    plot_filename = "confusion_matrix.png"
    plt.savefig(plot_filename)
    mlflow.log_artifact(plot_filename)

    print(f"Flat Run Finished. Accuracy: {acc:.4f} | Model logged: {model_filename}")