import os, sys
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from climateChange.loggers import logger
from climateChange.exceptions import ModelTrainerException
from climateChange.entity import ModelTrainerConfig, ModelTrainerArtifact
from climateChange.utils import save_object, save_json, evaluate_models


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def load_data(self) -> tuple:
        """Load Training and Testing Data"""
        try:
            logger.info(f"Loading Training and Testing Data")

            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            logger.info(f"Train Data Shape: {train_df.shape}")
            logger.info(f"Test Data Shape: {test_df.shape}")

            return train_df, test_df
        
        except Exception as e:
            logger.error(f"Error Loading Data: {e}")
            raise ModelTrainerException(e, sys)
    
    def prepare_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        """Prepare Data for Training"""
        try:
            logger.info("Prepare Data for Training")

            target_column = self.config.target_column

            X_train = train_df.drop([target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop([target_column], axis=1)
            y_test = test_df[target_column]

            logger.info(f"Feature shape: {X_train.shape}")
            logger.info(f"Target shape: {y_train.shape}")

            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logger.error(f"Error Preparing Data: {e}")
            raise ModelTrainerException(e, sys)
    
    def get_models(self) -> dict:
        """Get Models for Training"""
        try:
            models = {
                "RandomForestRegressor": RandomForestRegressor(),
                "LinearRegression": LinearRegression(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "XGBRegressor": xgb.XGBRFRegressor()
            }

            return models
        except Exception as e:
            logger.error(f"Error Getting Models: {e}")
            raise ModelTrainerException(e, sys)
    
    def train_best_model(self, X_train, X_test, y_train, y_test) -> tuple:
        """Train and Select The Best Model"""
        try:
            logger.info("Training Multiple Models to Find the best one")

            models = self.get_models()
            params = self.config.params

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(sorted([report['test_r2'] for report in model_report.values()]))

            best_model_name = list(model_report.keys())[
                list([report['test_r2'] for report in model_report.values()]).index(best_model_score)
            ]

            best_model = models[best_model_name]

            # Retrain Best Model with Best Parameters
            best_params = params[best_model_name]
            gs = GridSearchCV(best_model, best_params, cv=3, scoring='neg_mean_absolute_error')
            gs.fit(X_train, y_train)

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_metrics = {
                'mae': mean_absolute_error(y_train, y_train_pred),
                'mse': mean_squared_error(y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'r2_score': r2_score(y_train, y_train_pred)
            }
            
            test_metrics = {
                'mae': mean_absolute_error(y_test, y_test_pred),
                'mse': mean_squared_error(y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'r2_score': r2_score(y_test, y_test_pred)
            }
            
            logger.info(f"Best model: {best_model_name}")
            logger.info(f"Best model R2 score: {best_model_score}")
            logger.info(f"Train metrics: {train_metrics}")
            logger.info(f"Test metrics: {test_metrics}")

            return best_model, best_model_name, train_metrics, test_metrics, model_report
        
        except Exception as e:
            logger.error(f"Error Training Models: {e}")
            raise ModelTrainerException(e, sys)
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """Initiate Model Training Process"""
        try:
            logger.info("Starting Model Training Process")

            train_df, test_df = self.load_data()

            X_train, X_test, y_train, y_test = self.prepare_data(train_df, test_df)

            best_model, best_model_name, train_metrics, test_metrics, model_report = self.train_best_model(
                X_train, X_test, y_train, y_test
            )

            os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok= True)
            save_object(self.config.trained_model_path, best_model)

            # Save Metrics
            all_metrics = {
                'best_model': best_model_name,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'all_models_report': model_report
            }

            save_json(Path(self.config.model_metrics_path), all_metrics)

            logger.info("Model training completed successfully")
            
            return ModelTrainerArtifact(
                trained_model_path=self.config.trained_model_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics,
                model_accuracy=test_metrics['r2_score'],
                model_name=best_model_name,
                is_trained=True,
                message="Model training completed successfully"
            )
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise ModelTrainerException(e, sys)