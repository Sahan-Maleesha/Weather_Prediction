import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_weather_model():
    """Load the pre-trained weather model"""
    try:
        model = joblib.load('models/weather_model_logistic_regression_20250801_194620.pkl')
        model_info = joblib.load('models/model_info_20250801_194620.pkl')
        
        print("✅ Model loaded successfully!")
        print(f"Model: {model_info['model_name']}")
        print(f"Training Accuracy: {model_info['accuracy']:.4f}")
        
        return model, model_info
    
    except FileNotFoundError as e:
        print("❌ Error: Model files not found!")
        print(f"Error: {e}")
        return None, None
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

MODEL, MODEL_INFO = load_weather_model()

def predict_weather(temperature, rainfall, humidity, pressure, today_raining):
    """Predict tomorrow's weather based on today's conditions"""
    
    if MODEL is None or MODEL_INFO is None:
        return {"error": "Model not loaded. Please check model files."}
    
    try:
        # Create input data
        input_data = {
            'temperature': float(temperature),
            'rainfall': float(rainfall),
            'humidity': float(humidity),
            'pressure': float(pressure),
            'today_raining': int(today_raining)
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Add ALL engineered features that were used during training
        input_df['temp_humidity_ratio'] = input_df['temperature'] / (input_df['humidity'] + 1)
        input_df['pressure_deviation'] = abs(input_df['pressure'] - 1013)
        input_df['high_humidity'] = (input_df['humidity'] > 70).astype(int)
        input_df['low_pressure'] = (input_df['pressure'] < 1010).astype(int)
        
        # Add missing interaction features
        input_df['temp_pressure_interaction'] = input_df['temperature'] * input_df['pressure']
        input_df['rain_humidity_interaction'] = input_df['rainfall'] * input_df['humidity']
        
        # Select features in correct order
        X_new = input_df[MODEL_INFO['feature_columns']]
        
        # Make prediction
        prediction = MODEL.predict(X_new)[0]
        
        # Get probability if available
        if hasattr(MODEL, 'predict_proba'):
            probabilities = MODEL.predict_proba(X_new)[0]
            prob_no_rain = probabilities[0]
            prob_rain = probabilities[1]
            
            return {
                'prediction': 'Rain Expected Tomorrow' if prediction == 1 else 'No Rain Tomorrow',
                'rain_probability': round(prob_rain * 100, 2),
                'confidence': round(max(prob_no_rain, prob_rain) * 100, 2),
                'prediction_code': int(prediction)
            }
        else:
            return {
                'prediction': 'Rain Expected Tomorrow' if prediction == 1 else 'No Rain Tomorrow',
                'prediction_code': int(prediction)
            }
            
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

def interactive_prediction():
    """Interactive function to get user input and make prediction"""
    
    if MODEL is None or MODEL_INFO is None:
        print("❌ Model not loaded. Please check model files.")
        return
    
    print("\n🌦️  WEATHER PREDICTION - Tomorrow's Rain Forecast")
    print("="*50)
    
    try:
        print("\nEnter today's weather conditions:")
        temperature = float(input("🌡️  Temperature (°C): "))
        rainfall = float(input("🌧️  Rainfall (mm): "))
        humidity = float(input("💧 Humidity (%): "))
        pressure = float(input("📊 Pressure (hPa): "))
        today_raining = int(input("☔ Is it raining today? (1=Yes, 0=No): "))
        
        print("\n⏳ Making prediction...")
        
        result = predict_weather(temperature, rainfall, humidity, pressure, today_raining)
        
        if 'error' in result:
            print(f"❌ {result['error']}")
        else:
            print("\n🔮 PREDICTION RESULTS")
            print("="*30)
            print(f"📊 Prediction: {result['prediction']}")
            if 'rain_probability' in result:
                print(f"🌧️  Rain Probability: {result['rain_probability']}%")
                print(f"✅ Confidence: {result['confidence']}%")
            
    except ValueError:
        print("❌ Invalid input. Please enter numeric values.")
    except KeyboardInterrupt:
        print("\n👋 Prediction cancelled.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if MODEL is not None:
        interactive_prediction()