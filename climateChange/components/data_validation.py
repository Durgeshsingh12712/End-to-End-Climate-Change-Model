import os, sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats

from climateChange.loggers import logger
from climateChange.exceptions import DataValidationException
from climateChange.entity import DataValidationConfig, DataValidationArtifact
from climateChange.utils import save_json

class DataValidation:
    def __init__(self, config: DataValidationConfig) -> bool:
        self.config = config
    
    def validate_all_files_exist(self) -> bool:
        """Validate if all required files exist"""
        try:
            validation_status = True
            all_files = os.listdir(self.config.unzip_data_dir)

            required_files = ["climate_data.csv", "social_data.csv"]

            for file in required_files:
                if file not in all_files:
                    validation_status = False
                    logger.error(F"Required File {file} not found in {self.config.unzip_data_dir}")
                else:
                    logger.info(f"File {file} found successfully")
            return validation_status
        
        except Exception as e:
            logger.error(f"Error Validation files existance: {e}")
            raise DataValidationException(e, sys)
    
    def validate_data_schema(self, df:pd.DataFrame, expected_schema: Dict, data_type: str) -> Tuple[bool, List]:
        """Validate Data Schema against expected Schema"""
        try:
            logger.info(f"Validating {data_type} data schema")

            validation_status = True
            missing_columns = []

            # Check for missing columns
            for column in expected_schema.keys():
                if column not in df.columns:
                    validation_status = False
                    missing_columns.append(column)
                    logger.error(f"Missing Column: {column} in {data_type} data")
                else:
                    # Check data type
                    expected_dtype = expected_schema[column]
                    actual_dtype = str(df[column].dtype)

                    # More flixible types checking
                    if expected_dtype == 'datetime64[ns]':
                        try:
                            pd.to_datetime(df[column])
                            logger.info(f"Column {column} can be converted tp datetime")
                        except:
                            logger.warning(f"Column {column} cannot be converted to datetime")
                    elif expected_dtype == 'float64':
                        if not pd.api.types.is_numeric_dtype(df[column]):
                            logger.warning(f"Column {column} is not integer: {actual_dtype}")
                    elif expected_dtype == 'int64':
                        if not pd.api.types.is_integer_dtype(df[column]):
                            logger.warning(f"Column {column} is not integer: {actual_dtype}")

            if validation_status:
                logger.info(f"{data_type} Data Schema validation passed")
            else:
                logger.error(f"{data_type} Data Schema validation failed")
            
            return validation_status, missing_columns
        
        except Exception as e:
            logger.error(f"Error Validating {data_type} Data Schema: {e}")
            raise DataValidationException(e, sys)
    
    def detect_data_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
        """Detect Data Drift using statistical tests"""
        try:
            logger.info("Detecting Data Drift")

            drift_detected = False
            numeric_columns = reference_df.select_dtypes(include=[np.number]).columns

            for column in numeric_columns:
                if column in current_df.columns:
                    # Kolmogorov-Smirnov test
                    ref_data = reference_df[column].dropna()
                    curr_data = current_df[column].dropna()

                    if len(ref_data) > 0 and len(curr_data) > 0:
                        ks_statistic, p_value = stats.ks_2samp(ref_data, curr_data)

                        if p_value < 0.05:
                            drift_detected = True
                            logger.warning(f"Data Drift Detect in column {column}: p-value = {p_value:.4f}")
                        else:
                            logger.info(f"No significant drift in column {column}: p-value = {p_value:.4f}")
            
            return drift_detected
        except Exception as e:
            logger.error(F"Error Detecting Data Drift: {e}")
            return False
        
    def make_json_serialization(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self.make_json_serialization(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return {self.make_json_serialization(item) for item in obj}
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif pd.api.types.is_integer_dtype(type(obj)):
            return int(obj)
        elif pd.api.types.is_float_dtype(type(obj)):
            return float(obj)
        else:
            return obj
        
    def validate_data_quality(self, df: pd.DataFrame, data_type: str) -> Dict:
        """Validate Data Quality Metrics"""
        try:
            logger.info(f"Validating {data_type} Data Quality")

            quality_report = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'data_types': df.dtypes.astype(str).to_dict()
            }

            # Check for critical quality issues
            missing_percentage = (df.isnull().sum() / len(df) * 100).to_dict()
            quality_report['missing_percentage'] = missing_percentage

            # Flag columns with high missing values
            high_missing_cols = [col for col, pct in missing_percentage.items() if pct> 50]
            if high_missing_cols:
                logger.warning(f"High Missing Values in columns: {high_missing_cols}")
                quality_report['high_missing_columns'] = high_missing_cols
            
            # Check for outliers in numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outliers_report = {}

            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5*IQR
                upper_bound = Q3 + 1.5*IQR

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outliers_report[col] = len(outliers)
            
            quality_report['outliers'] = outliers_report

            logger.info(f"{data_type} Data Quality Validation completed")
            return quality_report
        
        except Exception as e:
            logger.error(f"Error Validating {data_type} Data Quality: {e}")
            raise DataValidationException(e, sys)
        
    def initiate_date_validation(self) -> DataValidationArtifact:
        try:
            logger.info("Starting Data Validation Process")

            validation_status = True
            all_missing_columns = {}
            validation_reports = {}

            # Validate files existance
            file_exists = self.validate_all_files_exist()
            
            if not file_exists:
                logger.error(f"Required Files not found")
                validation_status = False
            else:
                climate_data_path = Path(self.config.unzip_data_dir) / "climate_data.csv"
                socila_data_path = Path(self.config.unzip_data_dir) / "social_data.csv"

                climate_df = pd.read_csv(climate_data_path)
                social_df = pd.read_csv(socila_data_path)

                # Validate climate Data Schema
                climate_validation, climate_missing = self.validate_data_schema(
                    climate_df, self.config.climate_schema, "climate"
                )
                
                social_validation, social_missing = self.validate_data_schema(
                    social_df, self.config.social_schema, "social"
                )

                #Overall Validation Status
                schema_validation = climate_validation and social_validation

                if not schema_validation:
                    validation_status = False
                    all_missing_columns = {
                        'climate':climate_missing,
                        'social': social_missing
                    }
                
                # Data Quality Validation
                climate_quality = self.validate_data_quality(climate_df, "climate")
                social_quality = self.validate_data_quality(social_df, "social")

                validation_reports = {
                    'climate_quality': climate_quality,
                    'social_quality': social_quality
                }

                # Data Drift Detection 
                if len(climate_df) > 10:
                    mid_point = len(climate_df) // 2
                    reference_climate = climate_df.iloc[:mid_point]
                    current_climate = climate_df.iloc[mid_point:]

                    drift_detected = self.detect_data_drift(reference_climate, current_climate)
                else:
                    drift_detected = False
            
            # Save Validation Status
            validation_status_dict = {
                'files_exist': file_exists,
                'schema_validation': schema_validation if file_exists else False,
                'climate_validation': climate_validation if file_exists else False,
                'social_validation': social_validation if file_exists else False,
                'data_drift': drift_detected if file_exists else False,
                'missing_columns': all_missing_columns,
                'validation_reports': validation_reports
            }

            os.makedirs(os.path.dirname(self.config.status_file), exist_ok=True)
            with open(self.config.status_file, 'w') as f:
                f.write(f"Validation status: {validation_status}\n")
                f.write(f"Files exists: {file_exists}\n")
                f.write(f"Schema Validation: {schema_validation if file_exists else False}\n")
                f.write(f"Data Drift Detected: {drift_detected if file_exists else False}\n")
            
            validation_status_dict = self.make_json_serialization(validation_status_dict)

            validation_report_path = Path(self.config.root_dir) / "validation_report.json"
            save_json(validation_report_path, validation_status_dict)

            return DataValidationArtifact(
                validation_status=validation_status,
                climate_data_validation_status=climate_validation if file_exists else False,
                social_data_validation_status=social_validation if file_exists else False,
                missing_columns=all_missing_columns,

                data_drift_status=drift_detected if file_exists else False,
                validation_report_path=validation_report_path,
                message=" Data Validaton Completed"
            )
        except Exception as e:
            logger.error(f"Error in Data Validation: {e}")
            raise DataValidationException(e, sys)