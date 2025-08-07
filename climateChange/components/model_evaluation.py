import os, sys
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from climateChange.loggers import logger
from climateChange.exceptions import ModelEvaluationException
from climateChange.entity import ModelEvaluationConfig, ModelEvaluationArtifact
from climateChange.utils import load_object, save_json

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def load_data_and_model(self) -> tuple:
        """Load test data and trained model"""
        try:
            logger.info("Loading test data and trained model")
            
            test_df = pd.read_csv(self.config.test_data_path)
            
            model = load_object(self.config.model_path)
            preprocessor = load_object(self.config.preprocessor_path)
            
            logger.info("Data and model loaded successfully")
            return test_df, model, preprocessor
            
        except Exception as e:
            logger.error(f"Error loading data and model: {e}")
            raise ModelEvaluationException(e, sys)

    def evaluate_model(self, model, X_test, y_test) -> dict:
        """Evaluate model performance"""
        try:
            logger.info("Evaluating model performance")
            
            y_pred = model.predict(X_test)
            
            metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2_score': r2_score(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            }
            
            logger.info(f"Model evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise ModelEvaluationException(e, sys)
    
    def compare_with_baseline(self, current_metrics: dict) -> tuple:
        """Compare current model with baseline"""
        try:
            logger.info("Comparing with baseline model")
            
            baseline_r2 = 0.0  # Baseline model
            threshold = 0.6    # Minimum acceptable R2 score
            
            current_r2 = current_metrics['r2_score']
            
            is_model_accepted = current_r2 > threshold
            improved_accuracy = current_r2 - baseline_r2
            
            logger.info(f"Current R2: {current_r2}")
            logger.info(f"Baseline R2: {baseline_r2}")
            logger.info(f"Improvement: {improved_accuracy}")
            logger.info(f"Model accepted: {is_model_accepted}")
            
            return is_model_accepted, improved_accuracy
            
        except Exception as e:
            logger.error(f"Error comparing with baseline: {e}")
            raise ModelEvaluationException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """Initiate model evaluation process"""
        try:
            logger.info("Starting model evaluation process")
            
            test_df, model, preprocessor = self.load_data_and_model()
            
            target_column = self.config.target_column
            X_test = test_df.drop([target_column], axis=1)
            y_test = test_df[target_column]
            
            current_metrics = self.evaluate_model(model, X_test, y_test)

            is_model_accepted, improved_accuracy = self.compare_with_baseline(current_metrics)
            
            evaluation_report = {
                'model_metrics': current_metrics,
                'is_model_accepted': is_model_accepted,
                'improved_accuracy': improved_accuracy,
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(self.config.metric_file_name), exist_ok=True)
            save_json(Path(self.config.metric_file_name), evaluation_report)
            
            logger.info("Model evaluation completed successfully")
            
            return ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_path=self.config.model_path,
                trained_model_path=self.config.model_path,
                train_model_metric_artifact=current_metrics,
                best_model_metric_artifact=current_metrics,
                evaluation_report_path=self.config.metric_file_name,
                difference=improved_accuracy,
                message="Model evaluation completed successfully"
            )
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            raise ModelEvaluationException(e, sys)