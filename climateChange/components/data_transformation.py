import os, sys, re
import pandas as pd
import numpy as np
from pathlib import Path
from textblob import TextBlob
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from climateChange.loggers import logger
from climateChange.exceptions import DataTransformationException
from climateChange.entity import DataTransformationConfig, DataTransformationArtifact
from climateChange.utils import save_object


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def load_data(self) -> tuple:
        """Load Climate and Social Media Datasets"""
        try:
            logger.info("Loading Datasets for Transformation")

            climate_path = Path(self.config.data_path) / "climate_data.csv"
            social_path = Path(self.config.data_path) / "social_data.csv"

            climate_df = pd.read_csv(climate_path)
            social_df = pd.read_csv(social_path)

            logger.info(f"Climate Data Shape: {climate_df.shape}")
            logger.info(f"Social Data Shape: {social_df.shape}")

            return climate_df, social_df
        except Exception as e:
            logger.error(f"Error Loading Data: {e}")
            raise DataTransformationException(e, sys)
    
    def preprocess_climate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Climate Data"""
        try:
            logger.info("Preprocessing Climate Data")

            df['date'] = pd.to_datetime(df['date'])

            # Validate and clean critical columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(df[col].median())
            
            # Create Time-Based Features
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['season'] = df['month'].map({
                12: 0, 1: 0, 2: 0,  # Winter
                3: 1, 4: 1, 5: 1,  # Spring
                6: 2, 7: 2, 8: 2,  # Summer
                9: 3, 10: 3, 11: 3 # Fall
            })

            # Cyclical encoding for seasonal patterns
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Create lagged features
            for lag in [1, 3, 6]:
                df[f'temperature_lag_{lag}'] = df['temperature_anomaly'].shift(lag)
                df[f'co2_lag_{lag}'] = df['co2_concentration'].shift(lag)
            
            # Create rolling averages
            for window in [3, 6, 12]:
                df[f'temperature_rolling_{window}'] = df['temperature_anomaly'].rolling(window=window).mean()
                df[f'co2_rolling_{window}'] = df['co2_concentration'].rolling(window=window).mean()
            
            # Rate of change features
            df['temperature_change'] = df['temperature_anomaly'].diff()
            df['co2_change'] = df['co2_concentration'].diff()

            df = df.bfill().ffill()

            logger.info(f"Climate Data Preprocessed. New ShapeL {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error Preprocessing Climate Data: {e}")
            raise DataTransformationException(e, sys)
    
    def preprocess_social_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Social Media Data"""
        try:
            logger.info(f"Preprocessing Social Media Data")

            df['date'] = pd.to_datetime(df['date'])

            # Clean Text Data
            def clean_text(text):
                if pd.isna(text):
                    return ""
                text = str(text).lower()
                text = re.sub(r'http\S+|wwww\S+', '', text)
                text = re.sub(r'@\w+|#\w+', '', text)
                text = re.sub(r'[^a-zA-Z\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            
            df['cleaned_text'] = df['text'].apply(clean_text)

            # Recalculate sentiment for cleaned Text
            def calculate_sentiment(text):
                if not text or len(text) < 3:
                    return 0.0
                try:
                    blob = TextBlob(text)
                    return blob.sentiment.polarity
                except:
                    return 0.0

            df['sentiment_calculated'] = df['cleaned_text'].apply(calculate_sentiment)
            df['sentiment'] = df['sentiment'].fillna(df['sentiment_calculated'])
            
            # Fill missing engagement metrics
            df['likesCount'] = pd.to_numeric(df['likesCount'], errors='coerce').fillna(0)
            df['commentsCount'] = pd.to_numeric(df['commentsCount'], errors='coerce').fillna(0)
            
            # Create engagement features
            df['total_engagement'] = df['likesCount'] + df['commentsCount']
            df['engagement_ratio'] = df['commentsCount'] / (df['likesCount'] + 1)
            
            # Text features
            df['text_length'] = df['cleaned_text'].str.len()
            df['word_count'] = df['cleaned_text'].str.split().str.len()
            
            # Create sentiment categories
            df['sentiment_category'] = pd.cut(df['sentiment'], 
                                            bins=[-1, -0.1, 0.1, 1], 
                                            labels=['negative', 'neutral', 'positive'])
            
            logger.info(f"Social data preprocessed. New shape: {df.shape}")
            return df
        
        except Exception as e:
            logger.error(f"Error Preprocessing Social Data: {e}")
            raise DataTransformationException(e, sys)
    
    def merge_datasets(self, climate_df: pd.DataFrame, social_df: pd.DataFrame) -> pd.DataFrame:
        """Merge climate and social media datasets"""
        try:
            logger.info("Merging climate and social media datasets")
            
            # Aggregate social data by month
            social_df['year_month'] = social_df['date'].dt.to_period('M')
            
            social_agg = social_df.groupby('year_month').agg({
                'sentiment': ['mean', 'std', 'count'],
                'likesCount': ['mean', 'sum'],
                'commentsCount': ['mean', 'sum'],
                'total_engagement': ['mean', 'sum'],
                'engagement_ratio': 'mean',
                'text_length': 'mean',
                'word_count': 'mean'
            }).reset_index()
            
            # Flatten column names
            social_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in social_agg.columns]
            social_agg = social_agg.rename(columns={'year_month_': 'year_month'})
            
            # Convert period to datetime for merging
            social_agg['date'] = social_agg['year_month'].dt.to_timestamp()
            
            # Aggregate climate data by month
            climate_df['year_month'] = climate_df['date'].dt.to_period('M')
            climate_monthly = climate_df.groupby('year_month').agg({
                'temperature_anomaly': 'mean',
                'precipitation_anomaly': 'mean',
                'co2_concentration': 'mean',
                'sea_level_change': 'mean',
                'solar_radiation': 'mean',
                'year': 'first',
                'month': 'first',
                'quarter': 'first',
                'season': 'first',
                'month_sin': 'first',
                'month_cos': 'first'
            }).reset_index()
            
            climate_monthly['date'] = climate_monthly['year_month'].dt.to_timestamp()
            
            # Merge datasets
            merged_df = pd.merge(climate_monthly, social_agg, on='date', how='left')
            
            # Fill missing social media values
            social_columns = [col for col in merged_df.columns if any(x in col for x in ['sentiment', 'likes', 'comments', 'engagement', 'text', 'word'])]
            for col in social_columns:
                merged_df[col] = merged_df[col].fillna(merged_df[col].median() if merged_df[col].dtype in ['float64', 'int64'] else 0)
            
            # Drop unnecessary columns
            merged_df = merged_df.drop(['year_month_x', 'year_month_y'], axis=1, errors='ignore')
            
            logger.info(f"Datasets merged successfully. Shape: {merged_df.shape}")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
            raise DataTransformationException(e, sys)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for modeling"""
        try:
            logger.info("Creating additional features")
            
            # Interaction features
            if 'sentiment_mean' in df.columns:
                df['temp_sentiment_interaction'] = df['temperature_anomaly'] * df['sentiment_mean']
                df['co2_sentiment_interaction'] = df['co2_concentration'] * df['sentiment_mean']
            
            # Climate extremes indicators
            df['temp_extreme'] = (np.abs(df['temperature_anomaly']) > df['temperature_anomaly'].std() * 2).astype(int)
            df['co2_high'] = (df['co2_concentration'] > df['co2_concentration'].quantile(0.8)).astype(int)
            
            # Robust trend calculation using scipy
            def calculate_trend_slope(series, window=6):
                def trend_slope(x):
                    # Check for sufficient data points
                    if len(x) < 2:
                        return np.nan
                    
                    # Check for NaN values
                    if x.isnull().any():
                        return np.nan
                    
                    # Check for constant values (would cause issues)
                    if len(set(x)) <= 1:
                        return 0.0
                    
                    try:
                        slope, _, _, _, _ = stats.linregress(range(len(x)), x)
                        return slope
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                        return np.nan
                
                return series.rolling(window=window, min_periods=2).apply(trend_slope)
            
            # Calculate trends using robust method
            df['temp_trend'] = calculate_trend_slope(df['temperature_anomaly'])
            df['co2_trend'] = calculate_trend_slope(df['co2_concentration'])
            
            # Fill any remaining NaN values
            df = df.bfill().ffill().fillna(0)
            
            logger.info(f"Feature creation completed. Final shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            raise DataTransformationException(e, sys)
        
    def get_data_transformer_object(self, feature_columns) -> ColumnTransformer:
        """Create Data Transformation Pipeline"""
        try:
            logger.info(f"Creating Data Transformation object")

            numeric_features = feature_columns

            numeric_transformer = StandardScaler()

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features)
                ],
                remainder='drop'
            )

            logger.info(f"Data Transformer Object Created Successfully")
            return preprocessor
        
        except Exception as e:
            logger.error(f"Error Creating Data Transformer: {e}")
            raise DataTransformationException(e, sys)
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """Initiate Data Transformation Process"""
        try:
            logger.info(f"Staring Data Transformation Process")

            climate_df, social_df = self.load_data()

            climate_preprossed = self.preprocess_climate_data(climate_df)
            social_preprossed = self.preprocess_social_data(social_df)

            merged_df = self.merge_datasets(climate_preprossed, social_preprossed)

            final_df = self.create_features(merged_df)

            target_column = 'temperature_anomaly'
            X = final_df.drop([target_column, 'date'], axis=1, errors='ignore')
            y = final_df[target_column]

            feature_names = X.columns.tolist()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                shuffle=False
            )

            preprocessor = self.get_data_transformer_object(feature_names)

            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            # Creat Final DataSets
            train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
            train_df[target_column] = y_train.values

            test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
            test_df[target_column] = y_test.values

            os.makedirs(os.path.dirname(self.config.transformed_train_path), exist_ok=True)
            train_df.to_csv(self.config.transformed_train_path, index=False)
            test_df.to_csv(self.config.transformed_test_path, index=False)
            
            save_object(preprocessor, Path(self.config.preprocessor_path))
            
            logger.info("Data transformation completed successfully")
            
            return DataTransformationArtifact(
                transformed_train_path=self.config.transformed_train_path,
                transformed_test_path=self.config.transformed_test_path,
                preprocessor_path=self.config.preprocessor_path,
                feature_names=feature_names,
                transformation_status=True,
                message="Data transformation completed successfully"
            )
            
        except Exception as e:
            logger.error(f"Error in data transformation: {e}")
            raise DataTransformationException(e, sys)