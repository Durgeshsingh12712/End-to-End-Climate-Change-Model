import math
from flask import Flask, request, render_template

from climateChange.loggers import logger
from climateChange.pipeline import CustomData, PredictionPipeline, PredictPipelineAlternative

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Get from Data
            precipitation_anomaly = float(request.form.get('precipitation_anomaly'))
            co2_concentration = float(request.form.get('co2_concentration'))
            sea_level_change = float(request.form.get('sea_level_change'))
            solar_radiation = float(request.form.get('solar_radiation'))
            temperature_anomaly = float(request.form.get('temperature_anomaly'))
            year = int(request.form.get('year'))
            month = int(request.form.get('month'))

            # Calculate Derived Features
            quarter = (month - 1) // 3 + 1
            season = ((month - 1 ) // 3 + 1) % 4
            month_sin = math.sin(2 * math.pi * month / 12)
            month_cos = math.cos(2 * math.pi * month / 12)

            # Optional Social Media Features
            sentiment_mean = float(request.form.get('sentiment_mean', 0.0))
            sentiment_std = float(request.form.get('sentiment_std', 0.1))
            sentiment_count = int(request.form.get('sentiment_count', 100))
            likesCount_mean = float(request.form.get('likesCount_mean', 10.0))
            likesCount_sum = float(request.form.get('likesCount_sum', 1000.0))
            commentsCount_mean = float(request.form.get('commentsCount_mean', 5.0))
            commentsCount_sum = float(request.form.get('commentsCount_sum', 500.0))
            total_engagement_mean = float(request.form.get('total_engagement_mean', 15.0))
            engagement_ratio_mean = float(request.form.get('engagement_ratio_mean', 0.5))
            text_length_mean = float(request.form.get('text_length_mean', 100.0))
            word_count_mean = float(request.form.get('word_count_mean', 20.0))

            # Create CustomData object
            data = CustomData(
                precipitation_anomaly=precipitation_anomaly,
                co2_concentration=co2_concentration,
                sea_level_change=sea_level_change,
                solar_radiation=solar_radiation,
                temperature_anomaly=temperature_anomaly,
                year=year,
                month=month,
                quarter=quarter,
                season=season,
                month_sin=month_sin,
                month_cos=month_cos,
                sentiment_mean=sentiment_mean,
                sentiment_std=sentiment_std,
                sentiment_count=sentiment_count,
                likesCount_mean=likesCount_mean,
                likesCount_sum=likesCount_sum,
                commentsCount_mean=commentsCount_mean,
                commentsCount_sum=commentsCount_sum,
                total_engagement_mean=total_engagement_mean,
                engagement_ratio_mean=engagement_ratio_mean,
                text_length_mean=text_length_mean,
                word_count_mean=word_count_mean
            )

            pred_df = data.get_data_as_data_frame()

            try:
                predict_pipeline = PredictionPipeline()
                results = predict_pipeline.predict(pred_df)
            except Exception as ex:
                logger.warning(f"Standard Pipeline Failed: {ex}")
                if PredictPipelineAlternative:
                    logger.info("Trying Alternative Pipeline...")
                    predict_pipeline = PredictPipelineAlternative()
                    results =predict_pipeline.predict(pred_df)
                else:
                    raise ex
            
            logger.info(f"Prediction Successfully:; {results[0]}")

            return render_template('home.html', results=results[0])
        except Exception as e:
            logger.error(F"Error in Prediction:{e}")
            error_message = f"An error occured during Prediction: {str(e)}"
            return render_template('home.html', error=error_message)
        
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/app/predict', methods=['POST'])
def api_predict():
    """Api Endpoint For Programmatic Access"""
    try:
        data = request.get_json()

        # Validate Required fields
        required_fields = ['precipitation_anomaly', 'co2_concentration', 'sea_level_change', 
                          'solar_radiation', 'temperature_anomaly', 'year', 'month']
        
        for field in required_fields:
            if field not in data:
                return {'error': f'Missing required field: {field}'}, 400
            
        # Calculate Derived Features
        month = data['month']
        quarter = (month - 1) // 3 + 1
        season = ((month - 1) // 3 + 1) % 4
        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)
        
        # Create CustomData object with defaults for optional fields
        custom_data = CustomData(
            precipitation_anomaly=data['precipitation_anomaly'],
            co2_concentration=data['co2_concentration'],
            sea_level_change=data['sea_level_change'],
            solar_radiation=data['solar_radiation'],
            temperature_anomaly=data['temperature_anomaly'],
            year=data['year'],
            month=month,
            quarter=quarter,
            season=season,
            month_sin=month_sin,
            month_cos=month_cos,
            sentiment_mean=data.get('sentiment_mean', 0.0),
            sentiment_std=data.get('sentiment_std', 0.1),
            sentiment_count=data.get('sentiment_count', 100),
            likesCount_mean=data.get('likesCount_mean', 10.0),
            likesCount_sum=data.get('likesCount_sum', 1000.0),
            commentsCount_mean=data.get('commentsCount_mean', 5.0),
            commentsCount_sum=data.get('commentsCount_sum', 500.0),
            total_engagement_mean=data.get('total_engagement_mean', 15.0),
            engagement_ratio_mean=data.get('engagement_ratio_mean', 0.5),
            text_length_mean=data.get('text_length_mean', 100.0),
            word_count_mean=data.get('word_count_mean', 20.0)
        )

        pred_df = custom_data.get_data_as_data_frame()

        try:
            predict_pipeline = PredictionPipeline()
            results = predict_pipeline.predict(pred_df)
        except Exception as ex:
            logger.warning(f"Standard Pipeline Failed: {ex}")
            if PredictPipelineAlternative:
                logger.info("Trying Alternative Pipeline")
                predict_pipeline = PredictPipelineAlternative()
                results = predict_pipeline.predict(pred_df)
            else:
                raise ex
        
        return {
            'prediction': float(results[0]),
            'status': 'success'
        }
    except Exception as e:
        logger.error(f"API Prediction Error: {e}")
        return {'error': str(e), 'status':'error'}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)