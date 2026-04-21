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
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("2_Nested_Pipeline")

# Start the main parent run for the entire pipeline
with mlflow.start_run(run_name="Full_Pipeline_Run"):
    # --- STAGE 1: DATA PREPARATION ---
    with mlflow.start_run(run_name="Stage_1_Data_Prep", nested=True):
        local_path = config['data']['local_path']

        # Check if dataset exists locally; if not, download it
        if os.path.exists(local_path):
            mnist = joblib.load(local_path)
        else:
            mnist = fetch_openml(config['data']['dataset_name'], version=1, as_frame=False, parser='auto')
            joblib.dump(mnist, local_path)

        # Split dataset into training and testing sets using config parameters
        X_train, X_test, y_train, y_test = train_test_split(
            mnist.data, mnist.target,
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state']
        )

        # Log data-related parameters to the current nested run
        mlflow.log_params(config['data'])
        print("Stage 1: Data Ready")

    # --- STAGE 2: TRAINING  ---
    with mlflow.start_run(run_name="Stage_2_Training", nested=True):
        # Initialize and train the Random Forest model
        model = RandomForestClassifier(
            n_estimators=config['model']['n_estimators'],
            max_depth=config['model']['max_depth'],
            n_jobs=config['model']['n_jobs'],
            random_state=config['data']['random_state']
        )
        model.fit(X_train, y_train)

        # Log model hyperparameters
        mlflow.log_params(config['model'])

        # Save the model locally and log it as an artifact
        model_name = "nested_model.pkl"
        joblib.dump(model, model_name)
        mlflow.log_artifact(model_name)
        print("Stage 2: Model Trained and Logged")

    # --- STAGE 3: EVALUATION ---
    with mlflow.start_run(run_name="Stage_3_Evaluation", nested=True):
        # Predict on test data and calculate accuracy
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log the accuracy metric
        mlflow.log_metric("accuracy", acc)

        # Create and log the Confusion Matrix plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        plt.title(f"Nested Run Accuracy: {acc:.4f}")

        plot_name = "nested_confusion_matrix.png"
        plt.savefig(plot_name)
        mlflow.log_artifact(plot_name)
        plt.close()
        print(f"Stage 3: Evaluation Complete. Acc: {acc:.4f}")