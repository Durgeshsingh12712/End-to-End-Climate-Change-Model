import os, sys
import pandas as pd

from climateChange.exceptions import CCException
from climateChange.loggers import logger
from climateChange.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass

    def engineer_features(self, df):
        """Create engineered features expected by the model"""
        try:
            logger.info("Starting feature engineering")
            
            # Create trend features (using current values for single predictions)
            df['co2_trend'] = df['co2_concentration']
            df['temp_trend'] = df.get('temperature_anomaly', 0)
            
            # Create high/extreme indicators based on thresholds
            # Adjust these thresholds based on your training data statistics
            df['co2_high'] = (df['co2_concentration'] > 400).astype(int)
            df['temp_extreme'] = (df.get('temperature_anomaly', 0) > 1.0).astype(int)
            
            # Create interaction features
            df['temp_sentiment_interaction'] = df.get('temperature_anomaly', 0) * df['sentiment_mean']
            df['co2_sentiment_interaction'] = df['co2_concentration'] * df['sentiment_mean']
            
            # Create total engagement sum
            df['total_engagement_sum'] = df['total_engagement_mean'] * df['sentiment_count']
            
            logger.info("Feature engineering completed successfully")
            return df
        except Exception as e:
            logger.error(f"Error in Feature engineering: {e}")
            raise CCException(e, sys)
    
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts/model_trainer/model.pkl")
            preprocessor_path = os.path.join("artifacts/data_transformation/preprocessor.pkl")

            logger.info("Loading Model and Preprocessor")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logger.info("Features Engineering")
            features_engineered = self.engineer_features(features.copy())

            logger.info(f"Features after engineering: {list(features_engineered.columns)}")
            logger.info(f"Faetures shape after engineering: {features_engineered.shape}")

            logger.info(f"Scaling input features with full feature set")
            data_scaled = preprocessor.transform(features_engineered)

            logger.info("Making Prediction")
            preds = model.predict(data_scaled)
            logger.info(f"Prediction Completed: {preds}")
            return preds
        
        except Exception as e:
            logger.error(f"Error in Prediction Pipeline: {e}")
            raise CCException(e, sys)

class CustomData:
    def __init__(self,
                 precipitation_anomaly: float,
                 co2_concentration: float,
                 sea_level_change: float,
                 solar_radiation: float,
                 temperature_anomaly: float,  # Added this required field
                 year: int,
                 month: int,
                 quarter: int,
                 season: int,
                 month_sin: float,
                 month_cos: float,
                 sentiment_mean: float = 0.0,
                 sentiment_std: float = 0.1,
                 sentiment_count: int = 100,
                 likesCount_mean: float = 10.0,
                 likesCount_sum: float = 1000.0,
                 commentsCount_mean: float = 5.0,
                 commentsCount_sum: float = 500.0,
                 total_engagement_mean: float = 15.0,
                 engagement_ratio_mean: float = 0.5,
                 text_length_mean: float = 100.0,
                 word_count_mean: float = 20.0):
        
        self.precipitation_anomaly = precipitation_anomaly
        self.co2_concentration = co2_concentration
        self.sea_level_change = sea_level_change
        self.solar_radiation = solar_radiation
        self.temperature_anomaly = temperature_anomaly  # Store temperature anomaly
        self.year = year
        self.month = month
        self.quarter = quarter
        self.season = season
        self.month_sin = month_sin
        self.month_cos = month_cos
        self.sentiment_mean = sentiment_mean
        self.sentiment_std = sentiment_std
        self.sentiment_count = sentiment_count
        self.likesCount_mean = likesCount_mean
        self.likesCount_sum = likesCount_sum
        self.commentsCount_mean = commentsCount_mean
        self.commentsCount_sum = commentsCount_sum
        self.total_engagement_mean = total_engagement_mean
        self.engagement_ratio_mean = engagement_ratio_mean
        self.text_length_mean = text_length_mean
        self.word_count_mean = word_count_mean
    
    def get_data_as_data_frame(self):
        try:
            logger.info("Creating dataframe from custom data")
            
            custom_data_input_dict = {
                "precipitation_anomaly": [self.precipitation_anomaly],
                "co2_concentration": [self.co2_concentration],
                "sea_level_change": [self.sea_level_change],
                "solar_radiation": [self.solar_radiation],
                "temperature_anomaly": [self.temperature_anomaly],  # Include temperature
                "year": [self.year],
                "month": [self.month],
                "quarter": [self.quarter],
                "season": [self.season],
                "month_sin": [self.month_sin],
                "month_cos": [self.month_cos],
                "sentiment_mean": [self.sentiment_mean],
                "sentiment_std": [self.sentiment_std],
                "sentiment_count": [self.sentiment_count],
                "likesCount_mean": [self.likesCount_mean],
                "likesCount_sum": [self.likesCount_sum],
                "commentsCount_mean": [self.commentsCount_mean],
                "commentsCount_sum": [self.commentsCount_sum],
                "total_engagement_mean": [self.total_engagement_mean],
                "engagement_ratio_mean": [self.engagement_ratio_mean],
                "text_length_mean": [self.text_length_mean],
                "word_count_mean": [self.word_count_mean]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logger.info(f"Created dataframe with shape: {df.shape}")
            logger.info(f"Dataframe columns: {list(df.columns)}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error creating dataframe: {e}")
            raise CCException(e, sys)

# Alternative approach if the preprocessor only expects final features
class PredictPipelineAlternative:
    def __init__(self):
        pass
    
    def engineer_features(self, df):
        """Create only the final features that the model expects"""
        try:
            logger.info("Starting feature engineering for final features only")
            
            # Create trend features (using current values for single predictions)
            co2_trend = df['co2_concentration'].iloc[0]
            temp_trend = df.get('temperature_anomaly', pd.Series([0])).iloc[0]
            
            # Create high/extreme indicators based on thresholds
            co2_high = int(df['co2_concentration'].iloc[0] > 400)
            temp_extreme = int(df.get('temperature_anomaly', pd.Series([0])).iloc[0] > 1.0)
            
            # Create interaction features
            temp_sentiment_interaction = df.get('temperature_anomaly', pd.Series([0])).iloc[0] * df['sentiment_mean'].iloc[0]
            co2_sentiment_interaction = df['co2_concentration'].iloc[0] * df['sentiment_mean'].iloc[0]
            
            # Create total engagement sum
            total_engagement_sum = df['total_engagement_mean'].iloc[0] * df['sentiment_count'].iloc[0]
            
            # Create final features dataframe with only the features the model expects
            final_features = pd.DataFrame({
                'co2_trend': [co2_trend],
                'temp_sentiment_interaction': [temp_sentiment_interaction],
                'co2_sentiment_interaction': [co2_sentiment_interaction],
                'temp_extreme': [temp_extreme],
                'co2_high': [co2_high],
                'total_engagement_sum': [total_engagement_sum],
                'temp_trend': [temp_trend]
            })
            
            logger.info("Feature engineering completed successfully")
            logger.info(f"Final features: {list(final_features.columns)}")
            return final_features
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise CCException(e, sys)
    
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model_trainer", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "data_transformation", "preprocessor.pkl")
            
            logger.info("Loading model and preprocessor")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            logger.info("Engineering features")
            features_final = self.engineer_features(features)
            
            logger.info(f"Final features shape: {features_final.shape}")
            logger.info(f"Final features columns: {list(features_final.columns)}")
            
            logger.info("Scaling input features")
            data_scaled = preprocessor.transform(features_final)
            
            logger.info("Making prediction")
            preds = model.predict(data_scaled)
            
            logger.info(f"Prediction completed: {preds}")
            return preds
        
        except Exception as e:
            logger.error(f"Error in prediction pipeline: {e}")
            raise CCException(e, sys)
