#Training Source code
import mlflow #MLflow is used for tracking the model training process
import logging #Logging is used for logging the model training process
from sklearn.datasets import load_iris #Iris dataset is used for training the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("mlops.log"), 
    logging.StreamHandler()
    ]
    )
logging.info("Starting the model training process...")

logging.info("Loading the dataset...")
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

logging.info("Training the random forest model...")
with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    mlflow.log_metric("accuracy", accuracy)
    logging.info("Model training completed.")  
    logging.info(f"Model accuracy: {accuracy:.4f}")  
