import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load model
@st.cache_resource
def load_weather_model():
    try:
        model = joblib.load('D:/Rain Predeiction/models/weather_model_logistic_regression_20250801_194620.pkl')
        model_info = joblib.load('D:/Rain Predeiction/models/model_info_20250801_194620.pkl')
        return model, model_info
    except FileNotFoundError:
        st.error("❌ Model files not found!")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

MODEL, MODEL_INFO = load_weather_model()

# Prediction function
def predict_weather(temperature, rainfall, humidity, pressure, today_raining):
    try:
        input_data = {
            'temperature': float(temperature),
            'rainfall': float(rainfall),
            'humidity': float(humidity),
            'pressure': float(pressure),
            'today_raining': int(today_raining)
        }

        input_df = pd.DataFrame([input_data])

        # Feature engineering
        input_df['temp_humidity_ratio'] = input_df['temperature'] / (input_df['humidity'] + 1)
        input_df['pressure_deviation'] = abs(input_df['pressure'] - 1013)
        input_df['high_humidity'] = (input_df['humidity'] > 70).astype(int)
        input_df['low_pressure'] = (input_df['pressure'] < 1010).astype(int)
        input_df['temp_pressure_interaction'] = input_df['temperature'] * input_df['pressure']
        input_df['rain_humidity_interaction'] = input_df['rainfall'] * input_df['humidity']

        X_new = input_df[MODEL_INFO['feature_columns']]
        prediction = MODEL.predict(X_new)[0]

        if hasattr(MODEL, 'predict_proba'):
            prob = MODEL.predict_proba(X_new)[0]
            return {
                'prediction': 'Rain Expected Tomorrow' if prediction == 1 else 'No Rain Tomorrow',
                'rain_probability': round(prob[1] * 100, 2),
                'confidence': round(max(prob[0], prob[1]) * 100, 2),
                'prediction_code': int(prediction)
            }
        else:
            return {
                'prediction': 'Rain Expected Tomorrow' if prediction == 1 else 'No Rain Tomorrow',
                'prediction_code': int(prediction)
            }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# Streamlit UI
st.title("🌦️ Weather Forecast: Will It Rain Tomorrow?")
st.markdown("Provide today's weather conditions to predict if it will rain tomorrow.")

if MODEL is None:
    st.warning("Model not loaded. Please check the files in the 'models' directory.")
else:
    with st.form("weather_form"):
        temperature = st.number_input("🌡️ Temperature (°C)", min_value=-50.0, max_value=60.0, value=25.0)
        rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=500.0, value=5.0)
        humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
        pressure = st.number_input("📊 Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1013.0)
        today_raining = st.radio("☔ Is it raining today?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        with st.spinner("Making prediction..."):
            result = predict_weather(temperature, rainfall, humidity, pressure, today_raining)

        if 'error' in result:
            st.error(result['error'])
        else:
            st.success(result['prediction'])
            st.metric("🌧️ Rain Probability", f"{result['rain_probability']}%")
            st.metric("✅ Confidence", f"{result['confidence']}%")
