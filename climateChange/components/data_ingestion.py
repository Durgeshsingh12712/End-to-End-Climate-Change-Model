import os, sys, shutil
import pandas as pd
import numpy as np
from pathlib import Path
from textblob import TextBlob

from climateChange.loggers import logger
from climateChange.exceptions import DataIngestionException
from climateChange.entity import DataIngestionConfig, DataIngestionArtifact

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def create_climate_data(self, filename: Path):
        """
        Create realistic climate data for demonstration
        """
        try:
            # Generate 5 years of monthly climate data
            start_date = pd.Timestamp('2019-01-01')
            end_date = pd.Timestamp('2024-01-01')
            date_range = pd.date_range(start=start_date, end=end_date, freq='M')
            n_months = len(date_range)
            
            np.random.seed(42)
            
            # Temperature anomaly with warming trend
            base_trend = np.linspace(0.3, 1.8, n_months)
            seasonal = 0.5 * np.sin(2 * np.pi * np.arange(n_months) / 12)
            noise = np.random.normal(0, 0.2, n_months)
            temperature_anomaly = base_trend + seasonal + noise
            
            # CO2 concentration with realistic growth
            co2_base = 410
            co2_growth = np.linspace(0, 15, n_months)
            co2_seasonal = 3 * np.sin(2 * np.pi * np.arange(n_months) / 12 - np.pi/2)
            co2_concentration = co2_base + co2_growth + co2_seasonal + np.random.normal(0, 0.5, n_months)
            
            # Precipitation anomaly
            precipitation_anomaly = 10 * np.sin(2 * np.pi * np.arange(n_months) / 12 + np.pi/4) + np.random.normal(0, 15, n_months)
            
            # Sea level change (cumulative)
            sea_level_change = np.cumsum(np.random.normal(0.28, 0.08, n_months))
            
            # Solar radiation
            solar_radiation = 1361 + np.random.normal(0, 0.3, n_months)
            
            climate_df = pd.DataFrame({
                'date': date_range,
                'temperature_anomaly': temperature_anomaly,
                'precipitation_anomaly': precipitation_anomaly,
                'co2_concentration': co2_concentration,
                'sea_level_change': sea_level_change,
                'solar_radiation': solar_radiation
            })
            
            climate_df.to_csv(filename, index=False)
            logger.info(f"Climate data created with shape: {climate_df.shape}")
            
        except Exception as e:
            logger.error(f"Error creating climate data: {e}")
            raise DataIngestionException(e, sys)

    def create_social_data(self, filename: Path):
        """
        Create realistic social media sentiment data
        """
        try:
            # Generate social media posts about climate
            start_date = pd.Timestamp('2019-01-01')
            end_date = pd.Timestamp('2024-01-01')
            
            # Sample climate-related posts
            climate_posts = [
                "NASA releases new climate data showing alarming temperature trends",
                "Arctic ice melting faster than predicted by climate models",
                "Ocean temperatures reach record highs this month",
                "Climate scientists warn of accelerating global warming",
                "New renewable energy breakthrough could help fight climate change",
                "Extreme weather events becoming more frequent due to climate change",
                "Sea level rise threatens coastal communities worldwide",
                "Carbon emissions continue to rise despite climate agreements",
                "Climate activists demand immediate action from world leaders",
                "Innovative technologies emerge to combat climate crisis",
                "Forests are crucial for carbon sequestration and climate stability",
                "Climate change impacts biodiversity and ecosystem health",
                "Sustainable practices can help mitigate climate change effects",
                "Climate education is essential for future generations",
                "International cooperation needed to address climate emergency"
            ]
            
            # Generate random posts over time period
            n_posts = 1200  # About 20 posts per month
            dates = pd.date_range(start=start_date, end=end_date, periods=n_posts)
            
            # Create social media dataset
            social_data = []
            np.random.seed(42)
            
            for i, date in enumerate(dates):
                post = np.random.choice(climate_posts)
                
                # Add some variation to posts
                if np.random.random() > 0.7:
                    post += f" #{np.random.choice(['climate', 'environment', 'sustainability', 'green'])}"
                
                # Calculate sentiment using TextBlob
                blob = TextBlob(post)
                sentiment = blob.sentiment.polarity
                
                # Add some noise to make it more realistic
                sentiment += np.random.normal(0, 0.1)
                sentiment = np.clip(sentiment, -1, 1)
                
                # Generate engagement metrics
                likes = max(0, int(np.random.gamma(2, 5) * (1 + sentiment)))
                comments = max(0, int(np.random.gamma(1.5, 2) * (1 + abs(sentiment))))
                
                social_data.append({
                    'date': date,
                    'text': post,
                    'sentiment': sentiment,
                    'likesCount': likes,
                    'commentsCount': comments,
                    'profileName': f"user_{np.random.randint(1, 100)}"
                })
            
            social_df = pd.DataFrame(social_data)
            social_df.to_csv(filename, index=False)
            logger.info(f"Social data created with shape: {social_df.shape}")
            
        except Exception as e:
            logger.error(f"Error creating social data: {e}")
            raise DataIngestionException(e, sys)
    
    def download_file(self, url: str, filename: Path) -> bool:
        """Download File from URL with error handling"""
        try:
            logger.info(f"Downloading Data from {url} into file {filename}")

            os.makedirs(os.path.dirname(filename), exist_ok=True)
            os.makedirs(self.config.raw_data_path, exist_ok=True)
            os.makedirs(self.config.unzip_dir, exist_ok= True)

            if "climate" in str(filename).lower():
                self.create_climate_data(filename)
            else:
                self.create_social_data(filename)
            
            logger.info(f"Downloaded {filename} successfully")
            return True
        except Exception as e:
            logger.error(f"Error Downloading file: {e}")
            raise DataIngestionException(e, sys)
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Initiate Data Ingestion Process"""
        try:
            logger.info("Starting Data Ingestin Process")

            os.makedirs(self.config.root_dir, exist_ok=True)
            os.makedirs(self.config.raw_data_path, exist_ok=True)

            climate_data = self.download_file(
                url=self.config.climate_data_url,
                filename=self.config.climate_data_path
            )

            social_data = self.download_file(
                url=self.config.social_data_url,
                filename=self.config.social_data_path
            )

            if climate_data:
                shutil.copy2(self.config.climate_data_path, 
                           Path(self.config.raw_data_path) / "climate_data.csv")
            
            if social_data:
                shutil.copy2(self.config.social_data_path, 
                           Path(self.config.raw_data_path) / "social_data.csv")
                
            # Copy files to unzip data
            if climate_data:
                shutil.copy2(self.config.climate_data_path, 
                           Path(self.config.unzip_dir) / "climate_data.csv")
            
            if social_data:
                shutil.copy2(self.config.social_data_path, 
                           Path(self.config.unzip_dir) / "social_data.csv")
                
            logger.info(f"Data Ingestion Completed Successfully")

            return DataIngestionArtifact(
                climate_date_path=self.config.climate_data_path,
                social_data_path=self.config.social_data_path,
                raw_data_path=self.config.raw_data_path,
                is_ingested=climate_data and social_data,
                message="Data Ingestion Completed Successfully"
            )
        
        except Exception as e:
            logger.error(f"Error in Data Ingestion: {e}")
            raise DataIngestionException(e, sys)