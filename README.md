# End-to-End-Climate-Change-Model

# 🌍 Climate Prediction System

An end-to-end MLOps pipeline for predicting climate anomalies using machine learning, combining climate data with social media sentiment analysis.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Pipeline Stages](#pipeline-stages)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Project Overview

The Climate Prediction System is a comprehensive machine learning solution that predicts temperature anomalies by analyzing:

- **Climate Data**: Temperature, precipitation, CO2 concentration, sea level changes, and solar radiation
- **Social Media Sentiment**: Public opinion and engagement metrics related to climate discussions
- **Temporal Features**: Seasonal patterns, trends, and cyclical variations

### Key Objectives

- 🎯 Predict temperature anomalies with high accuracy
- 📊 Combine multiple data sources for comprehensive analysis
- 🚀 Implement MLOps best practices for production deployment
- 📈 Monitor model performance and data drift
- 🌐 Provide user-friendly web interface for predictions

## ✨ Features

### Core Functionality
- **Multi-Source Data Integration**: Climate data + social media sentiment
- **Advanced Feature Engineering**: Lag features, rolling averages, seasonal decomposition
- **Ensemble Model Training**: Random Forest, XGBoost, Gradient Boosting
- **Real-time Predictions**: Web-based prediction interface
- **Model Monitoring**: Performance tracking and drift detection

### MLOps Pipeline
- **Automated Data Ingestion**: Scheduled data collection and validation
- **Data Quality Checks**: Schema validation, outlier detection, drift monitoring
- **Model Training & Evaluation**: Automated model selection and hyperparameter tuning
- **CI/CD Integration**: Automated testing and deployment

### Technical Stack
- **Backend**: Python, Flask, scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy, TextBlob
- **ML Ops**: Docker, GitHub Actions
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: AWS/Azure/GCP compatible

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Data Pipeline │    │   ML Pipeline   │
│                 │    │                 │    │                 │
│ • Climate API   │────│ • Ingestion     │────│ • Training      │
│ • Social Media  │    │ • Validation    │    │ • Evaluation    │
│ • Weather Data  │    │ • Transform     │    │ • Deployment    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   Model Store   │    │   Web App       │
│                 │    │                 │    │                 │
│ • Performance   │    │ • Versioning    │    │ • Predictions   │
│ • Data Drift    │    │ • Artifacts     │    │ • Visualization │
│ • Alerts        │    │ • Metadata      │    │ • API           │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Git
- Virtual environment tool (venv/conda)

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Durgeshsingh12712/End-to-End-Climate-Change-Model.git
   cd End-to-End-Climate-Change-Model
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv climate_env
   source climate_env/bin/activate  # On Windows: climate_env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Configuration**
   ```bash
   # Create necessary directories
   mkdir -p artifacts logs config
   
   # Copy example configuration
   cp config/config.example.yaml config/config.yaml
   cp config/params.example.yaml config/params.yaml
   ```

5. **Initialize the Project**
   ```bash
   python setup.py install
   ```

### Docker Installation (Alternative)

```bash
# Build Docker image
docker build -t End-to-End-Climate-Change-Model .

# Run container
docker run -p 5000:5000 End-to-End-Climate-Change-Model
```

## 📖 Usage

### Training the Model

1. **Run Complete Pipeline**
   ```bash
   python main.py
   ```

2. **Run Individual Stages**
   ```bash
   # Data ingestion
   python -m climateChange/components/data_ingestion
   
   # Data validation
   python -m climateChange/components/data_validation
   
   # Data transformation
   python -m climateChange/components/data_transformation
   
   # Model training
   python -m climateChange/components/model_trainer
   
   # Model evaluation
   python -m climateChange/components/model_evaluation
   ```

### Making Predictions

1. **Web Interface**
   ```bash
   python app.py
   # Open http://localhost:5000 in browser
   ```

2. **Python API**
   ```python
   from climateChange.pipeline.prediction_pipeline import PredictPipeline, CustomData
   
   # Create custom data
   data = CustomData(
       precipitation_anomaly=-2.5,
       co2_concentration=415.3,
       sea_level_change=3.4,
       solar_radiation=1361.2,
       year=2024,
       month=6,
       quarter=2,
       season=2,
       month_sin=0.5,
       month_cos=0.866
   )
   
   # Make prediction
   predict_pipeline = PredictPipeline()
   prediction = predict_pipeline.predict(data.get_data_as_data_frame())
   print(f"Predicted Temperature Anomaly: {prediction[0]:.3f}°C")
   ```

## 📁 Project Structure

```
climate_prediction/
├── 📂 src/climate_prediction/
│   ├── 📂 components/          # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   ├── 📂 configure/              # Configuration management
│   │   └── configuration.py
│   ├── 📂 entity/              # Data classes
│   │   ├── config_entity.py
│   │   └── artifact_entity.py
│   ├── 📂 pipeline/            # Pipeline stages
│   │   ├── trainining_pipeline.py
│   │   └── predict_pipeline.py
│   ├── 📂 utils/               # Utility functions
│   │   └── tools.py
│   ├── 📂 constants/           # Project constants
│   │   └──constant.py        
│   ├── 📂 loggers/             # Logging configuration
│   │   └──logger.py
│   ├── 📂 exception/           # Custom Climate CHange Exception
        └──CCException.py

├── 📂 config/                  # Configuration files
│   ├── config.yaml
│   └── params.yaml
├── 📂 templates/               # Web templates
│   ├── index.html
│   └── home.html
│   └── base.html
│   └── about.html
├── 📂 artifacts/               # Generated artifacts
├── 📂 logs/                    # Log files
├── 📄 app.py                   # Flask web application
├── 📄 main.py                  # Main training pipeline
├── 📄 requirements.txt         # Dependencies
├── 📄 setup.py                 # Package setup
├── 📄 Dockerfile              # Docker configuration
└── 📄 README.md               # This file
```

## ⚙️ Configuration

### config.yaml
```yaml
# Main configuration file
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  climate_data_url: https://climate.nasa.gov/api/temperature-data
  social_data_url: https://api.github.com/repos/nasa/climate-data

model_trainer:
  root_dir: artifacts/model_trainer
  model_name: climate_predictor
  
```

### params.yaml
```yaml
# Model parameters
TEST_SIZE: 0.2
RANDOM_STATE: 42
TARGET_COLUMN: temperature_anomaly

model_params:
  RandomForestRegressor:
    n_estimators: [50, 100, 200]
    max_depth: [5, 10, 15, null]
  XGBRegressor:
    n_estimators: [50, 100, 200]
    learning_rate: [0.05, 0.1, 0.2]
```

## 🔄 Pipeline Stages

### 1. Data Ingestion
- **Purpose**: Collect climate and social media data
- **Input**: External APIs, databases
- **Output**: Raw CSV files
- **Features**:
  - Automated data collection
  - Data format validation
  - Error handling and retry logic

### 2. Data Validation  
- **Purpose**: Ensure data quality and consistency
- **Input**: Raw data files
- **Output**: Validation report
- **Checks**:
  - Schema validation
  - Missing value analysis
  - Outlier detection
  - Data drift monitoring

### 3. Data Transformation
- **Purpose**: Feature engineering and preprocessing
- **Input**: Validated data
- **Output**: Processed features, preprocessor object
- **Operations**:
  - Time series feature creation
  - Sentiment analysis
  - Scaling and encoding
  - Train-test splitting

### 4. Model Training
- **Purpose**: Train and select best performing model
- **Input**: Processed features
- **Output**: Trained model, metrics
- **Models**:
  - Random Forest Regressor
  - XGBoost Regressor
  - Gradient Boosting Regressor
  - Linear Regression

### 5. Model Evaluation
- **Purpose**: Evaluate model performance and log metrics
- **Input**: Trained model, test data
- **Output**: Evaluation metrics
- **Metrics**:
  - Mean Absolute Error (MAE)
  - Root Mean Square Error (RMSE)
  - R² Score
  - Mean Absolute Percentage Error (MAPE)

## 📊 Model Performance

### Current Best Model: XGBoost Regressor

| Metric | Train | Test |
|--------|--------|------|
| MAE    | 0.145  | 0.167 |
| RMSE   | 0.198  | 0.234 |
| R²     | 0.887  | 0.854 |
| MAPE   | 8.2%   | 9.7%  |

### Feature Importance

1. **CO2 Concentration** (23.4%)
2. **Temperature Trend** (18.7%)
3. **Seasonal Features** (15.2%)
4. **Solar Radiation** (12.8%)
5. **Social Sentiment** (11.3%)

## 🔌 API Documentation

### Prediction Endpoint

```http
POST /predictdata
Content-Type: application/x-www-form-urlencoded

precipitation_anomaly=-2.5&co2_concentration=415.3&sea_level_change=3.4...
```

### Response Format

```json
{
  "prediction": 1.234,
  "status": "success",
  "model_version": "v1.0.0",
  "confidence": 0.89
}
```

### Web Interface

Access the web interface at `http://localhost:5000` for:
- Interactive prediction form
- Real-time results
- Visualization dashboard
- Model performance metrics

## 🚀 Deployment

### Local Development
```bash
python app.py
# Access at http://localhost:5000
```

### Production Deployment

#### Docker
```bash
docker build -t End-to-End-Climate-Change-Model .
docker run -p 5000:5000 -e PORT=5000 End-to-End-Climate-Change-Model
```

#### Cloud Platforms

**AWS ECS/Fargate**
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin
docker tag End-to-End-Climate-Change-Model:latest <account>.dkr.ecr.us-east-1.amazonaws.com/End-to-End-Climate-Change-Model:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/End-to-End-Climate-Change-Model:latest
```

**Google Cloud Run**
```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/End-to-End-Climate-Change-Model
gcloud run deploy --image gcr.io/PROJECT-ID/End-to-End-Climate-Change-Model --platform managed
```

### Environment Variables

```bash
# Required for production
export MODEL_REGISTRY_URI="your-model-registry"
export DATA_SOURCE_API_KEY="your-api-key"
```

## 🔧 Development

### Setting up Development Environment

1. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Pre-commit hooks**
   ```bash
   pre-commit install
   ```

3. **Run tests**
   ```bash
   pytest tests/ -v --cov=src/End-to-End-Climate-Change-Model
   ```

4. **Code formatting**
   ```bash
   black src/
   isort src/
   flake8 src/
   ```

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement changes following project structure
3. Add tests for new functionality
4. Update documentation
5. Submit pull request

## 📈 Monitoring & Maintenance

### Model Monitoring
- **Performance Tracking**: Automated metrics collection
- **Data Drift Detection**: Statistical tests for input distribution changes
- **Prediction Drift**: Monitor prediction distribution over time
- **Alerting**: Email/Slack notifications for issues

### Maintenance Tasks
- **Weekly**: Review model performance metrics
- **Monthly**: Check for data drift and retrain if needed
- **Quarterly**: Evaluate new features and model architectures

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Ways to Contribute
- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit code improvements
- 🧪 Add test cases

### Contribution Process
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA Climate Change APIs** for providing climate data
- **NOAA** for weather and atmospheric data
- **scikit-learn** community for machine learning tools
- **MLflow** for experiment tracking capabilities

## 📞 Support

- **Documentation**: [Project Wiki](https://github.com/Durgeshsingh12712/End-to-End-Climate-Change-Model/wiki)
- **Issues**: [GitHub Issues](https://github.com/Durgeshsingh12712/End-to-End-Climate-Change-Model/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Durgeshsingh12712/End-to-End-Climate-Change-Model/discussions)
- **Email**: durgeshsingh12712@gmail.com

---

<div align="center">

**🌍 Building a sustainable future through predictive analytics 🌍**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

</div>