import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. SETUP AND CONFIGURATION
# ============================================================================

# Create directories for saving models and plots
def setup_directories():
    """Create necessary directories"""
    directories = ['models', 'plots', 'data']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")

setup_directories()

# ============================================================================
# 2. CREATE SAMPLE DATASET (Replace with your actual data loading)
# ============================================================================

def create_sample_weather_data(n_samples=7000, save_to_csv=True):
    """Create sample weather data similar to your dataset"""
    np.random.seed(42)
    
    print("🔄 Generating sample weather data...")
    
    # Generate correlated weather features
    temperature = np.random.normal(25, 8, n_samples)  # 25°C average
    humidity = np.random.normal(60, 20, n_samples)    # 60% average
    humidity = np.clip(humidity, 0, 100)              # Keep humidity 0-100%
    pressure = np.random.normal(1013, 15, n_samples)  # 1013 hPa average
    rainfall = np.random.exponential(2, n_samples)    # Exponential distribution
    
    # Today's rain (influenced by humidity and rainfall)
    today_rain_prob = (humidity/100 + rainfall/10) / 2
    today_raining = np.random.binomial(1, np.clip(today_rain_prob, 0, 1), n_samples)
    
    # Tomorrow's rain (influenced by today's conditions + some randomness)
    tomorrow_rain_prob = (today_raining * 0.6 + humidity/150 + rainfall/15 + 
                         (pressure < 1010) * 0.3) / 2
    tomorrow_raining = np.random.binomial(1, np.clip(tomorrow_rain_prob, 0, 1), n_samples)
    
    data = pd.DataFrame({
        'temperature': np.round(temperature, 1),
        'rainfall': np.round(rainfall, 2),
        'humidity': np.round(humidity, 1),
        'pressure': np.round(pressure, 1),
        'today_raining': today_raining,
        'tomorrow_raining': tomorrow_raining
    })
    
    if save_to_csv:
        csv_path = 'data/weather_data.csv'
        data.to_csv(csv_path, index=False)
        print(f"✅ Sample data saved to: {csv_path}")
    
    return data

def load_weather_data(file_path=None):
    """Load weather data from CSV or create sample data"""
    if file_path and os.path.exists(file_path):
        print(f"📂 Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        print("✅ Data loaded successfully!")
    else:
        if file_path:
            print(f"⚠️ File not found: {file_path}")
        print("🔄 Creating sample data...")
        df = create_sample_weather_data(7000, save_to_csv=True)
    
    return df


df = load_weather_data('data/Train Data.xlsx')  # Use this for your actual data
# df = load_weather_data()  # This creates sample data

print("\n📊 Dataset Information:")
print(f"Shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
print("\n📋 First 5 rows:")
print(df.head())
print("\n🏷️ Data types:")
print(df.dtypes)
print("\n🎯 Target distribution:")
print(df['tomorrow_raining'].value_counts())
print(f"Rain percentage: {df['tomorrow_raining'].mean()*100:.1f}%")

# ============================================================================
# 3. DATA PREPROCESSING AND FEATURE ENGINEERING
# ============================================================================

def prepare_features(df):
    """Prepare features for training"""
    print("\n🔧 Engineering features...")
    df_processed = df.copy()
    
    # Feature engineering
    df_processed['temp_humidity_ratio'] = df_processed['temperature'] / (df_processed['humidity'] + 1)
    df_processed['pressure_deviation'] = abs(df_processed['pressure'] - 1013)
    df_processed['high_humidity'] = (df_processed['humidity'] > 70).astype(int)
    df_processed['low_pressure'] = (df_processed['pressure'] < 1010).astype(int)
    df_processed['temp_pressure_interaction'] = df_processed['temperature'] * df_processed['pressure'] / 1000
    df_processed['rain_humidity_interaction'] = df_processed['rainfall'] * df_processed['humidity'] / 100
    
    print("✅ Feature engineering completed!")
    return df_processed

def analyze_data(df):
    """Perform basic data analysis"""
    print("\n📈 Data Analysis:")
    print("-" * 50)
    
    # Basic statistics
    print("🔢 Numerical features statistics:")
    numerical_cols = ['temperature', 'rainfall', 'humidity', 'pressure']
    print(df[numerical_cols].describe())
    
    # Correlation with target
    print("\n🎯 Correlation with target (tomorrow_raining):")
    correlations = df.corr()['tomorrow_raining'].sort_values(key=abs, ascending=False)
    for feature, corr in correlations.items():
        if feature != 'tomorrow_raining':
            print(f"{feature:25}: {corr:+.3f}")

df_processed = prepare_features(df)
analyze_data(df_processed)

# Prepare features (X) and target (y)
feature_columns = ['temperature', 'rainfall', 'humidity', 'pressure', 'today_raining',
                  'temp_humidity_ratio', 'pressure_deviation', 'high_humidity', 
                  'low_pressure', 'temp_pressure_interaction', 'rain_humidity_interaction']

X = df_processed[feature_columns]
y = df_processed['tomorrow_raining']

print(f"\n📊 Final dataset shape:")
print(f"Features: {X.shape}")
print(f"Target: {y.shape}")

# ============================================================================
# 4. TRAIN MULTIPLE MODELS AND COMPARE
# ============================================================================

def train_and_evaluate_models(X, y, test_size=0.2, random_state=42):
    """Train multiple models and compare performance"""
    print("\n🚀 Starting model training...")
    print("=" * 60)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"📊 Data split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training rain percentage: {y_train.mean()*100:.1f}%")
    print(f"Test rain percentage: {y_test.mean()*100:.1f}%")
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state
        ),
        'Logistic Regression': LogisticRegression(
            random_state=random_state, 
            max_iter=1000,
            C=1.0
        )
    }
    
    # Train and evaluate models
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate accuracies
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Store results
        results[name] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'predictions': y_pred_test,
            'probabilities': y_pred_proba,
            'model': model
        }
        trained_models[name] = model
        
        print(f"✅ {name} completed!")
        print(f"   Training Accuracy: {train_accuracy:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        print(f"   Overfitting: {train_accuracy - test_accuracy:.4f}")
    
    return results, trained_models, X_train, X_test, y_train, y_test

# Train models
results, trained_models, X_train, X_test, y_train, y_test = train_and_evaluate_models(X, y)

# ============================================================================
# 5. DETAILED MODEL EVALUATION
# ============================================================================

def detailed_evaluation(results, X_test, y_test, feature_columns):
    """Perform detailed evaluation of all models"""
    print("\n📋 DETAILED MODEL EVALUATION")
    print("=" * 60)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['test_accuracy']
    
    print(f"🏆 Best Model: {best_model_name}")
    print(f"🎯 Best Test Accuracy: {best_accuracy:.4f}")
    
    # Detailed evaluation for each model
    for name, result in results.items():
        print(f"\n📊 {name} Results:")
        print("-" * 40)
        y_pred = result['predictions']
        
        print(f"Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"Training Accuracy: {result['train_accuracy']:.4f}")
        print(f"Overfitting Gap: {result['train_accuracy'] - result['test_accuracy']:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Rain', 'Rain']))
    
    return best_model_name, best_model, best_accuracy

best_model_name, best_model, best_accuracy = detailed_evaluation(results, X_test, y_test, feature_columns)

# ============================================================================
# 6. VISUALIZATION
# ============================================================================

def create_visualizations(results, y_test, feature_columns, best_model_name):
    """Create and save visualization plots"""
    print(f"\n📊 Creating visualizations...")
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Model Comparison
    plt.subplot(2, 3, 1)
    model_names = list(results.keys())
    accuracies = [results[name]['test_accuracy'] for name in model_names]
    colors = ['#2E8B57' if name == best_model_name else '#4682B4' for name in model_names]
    
    bars = plt.bar(model_names, accuracies, color=colors)
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Confusion Matrix for best model
    plt.subplot(2, 3, 2)
    y_pred_best = results[best_model_name]['predictions']
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Rain', 'Rain'], 
                yticklabels=['No Rain', 'Rain'])
    plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # 3. Feature Importance (for Random Forest)
    if best_model_name == 'Random Forest':
        plt.subplot(2, 3, 3)
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.title('Feature Importance', fontsize=14, fontweight='bold')
        plt.xlabel('Importance')
        plt.tight_layout()
    
    # 4. Training vs Test Accuracy
    plt.subplot(2, 3, 4)
    model_names = list(results.keys())
    train_accs = [results[name]['train_accuracy'] for name in model_names]
    test_accs = [results[name]['test_accuracy'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, train_accs, width, label='Training', alpha=0.8)
    plt.bar(x + width/2, test_accs, width, label='Test', alpha=0.8)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Training vs Test Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(x, model_names)
    plt.legend()
    plt.ylim(0, 1)
    
    # 5. Data distribution
    plt.subplot(2, 3, 5)
    rain_counts = pd.Series(y_test).value_counts()
    plt.pie(rain_counts.values, labels=['No Rain', 'Rain'], autopct='%1.1f%%', startangle=90)
    plt.title('Test Set Distribution', fontsize=14, fontweight='bold')
    
    # 6. Prediction Confidence Distribution (if available)
    if results[best_model_name]['probabilities'] is not None:
        plt.subplot(2, 3, 6)
        probs = results[best_model_name]['probabilities']
        confidence = np.maximum(probs, 1-probs)
        plt.hist(confidence, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.axvline(confidence.mean(), color='red', linestyle='--', 
                   label=f'Mean: {confidence.mean():.3f}')
        plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f'plots/weather_model_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plots saved to: {plot_path}")
    
    # Show plot
    plt.show()

create_visualizations(results, y_test, feature_columns, best_model_name)

# ============================================================================
# 7. SAVE THE MODEL
# ============================================================================

def save_model_and_info(best_model, best_model_name, best_accuracy, feature_columns):
    """Save the trained model and associated information"""
    print(f"\n💾 Saving model...")
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the best model
    model_filename = f'models/weather_model_{best_model_name.lower().replace(" ", "_")}_{timestamp}.pkl'
    joblib.dump(best_model, model_filename)
    
    # Save feature columns and model info
    model_info = {
        'model_name': best_model_name,
        'accuracy': best_accuracy,
        'feature_columns': feature_columns,
        'training_date': datetime.now().isoformat(),
        'model_filename': model_filename,
        'model_type': type(best_model).__name__
    }
    
    info_filename = f'models/model_info_{timestamp}.pkl'
    joblib.dump(model_info, info_filename)
    
    # Also save as JSON for readability
    import json
    json_filename = f'models/model_info_{timestamp}.json'
    with open(json_filename, 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    
    print(f"✅ Model saved successfully!")
    print(f"   Model file: {model_filename}")
    print(f"   Info file: {info_filename}")
    print(f"   JSON info: {json_filename}")
    
    return model_filename, info_filename, model_info

model_filename, info_filename, model_info = save_model_and_info(
    best_model, best_model_name, best_accuracy, feature_columns
)

# ============================================================================
# 8. MODEL USAGE FUNCTIONS
# ============================================================================

def load_and_predict(model_path, info_path, new_data_dict):
    """Load saved model and make predictions"""
    try:
        # Load model and info
        loaded_model = joblib.load(model_path)
        model_info = joblib.load(info_path)
        
        # Prepare input data
        input_df = pd.DataFrame([new_data_dict])
        
        # Add engineered features (same as training)
        input_df['temp_humidity_ratio'] = input_df['temperature'] / (input_df['humidity'] + 1)
        input_df['pressure_deviation'] = abs(input_df['pressure'] - 1013)
        input_df['high_humidity'] = (input_df['humidity'] > 70).astype(int)
        input_df['low_pressure'] = (input_df['pressure'] < 1010).astype(int)
        input_df['temp_pressure_interaction'] = input_df['temperature'] * input_df['pressure'] / 1000
        input_df['rain_humidity_interaction'] = input_df['rainfall'] * input_df['humidity'] / 100
        
        # Select features in correct order
        X_new = input_df[model_info['feature_columns']]
        
        # Make prediction
        prediction = loaded_model.predict(X_new)[0]
        
        # Get probability if available
        if hasattr(loaded_model, 'predict_proba'):
            probability = loaded_model.predict_proba(X_new)[0]
            prob_no_rain = probability[0]
            prob_rain = probability[1]
            
            return {
                'prediction': 'Rain Tomorrow' if prediction == 1 else 'No Rain Tomorrow',
                'confidence': f"{max(prob_no_rain, prob_rain)*100:.1f}%",
                'rain_probability': f"{prob_rain*100:.1f}%",
                'no_rain_probability': f"{prob_no_rain*100:.1f}%",
                'model_used': model_info['model_name'],
                'model_accuracy': f"{model_info['accuracy']:.1%}"
            }
        else:
            return {
                'prediction': 'Rain Tomorrow' if prediction == 1 else 'No Rain Tomorrow',
                'model_used': model_info['model_name'],
                'model_accuracy': f"{model_info['accuracy']:.1%}"
            }
            
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# ============================================================================
# 9. TEST THE SAVED MODEL
# ============================================================================

def test_saved_model():
    """Test the saved model with example data"""
    print(f"\n🧪 Testing saved model...")
    print("=" * 50)
    
    test_cases = [
        {
            'name': 'Heavy Rain Conditions',
            'data': {'temperature': 24.0, 'rainfall': 8.5, 'humidity': 90.0, 'pressure': 1005.0, 'today_raining': 1}
        },
        {
            'name': 'Clear Weather Conditions',
            'data': {'temperature': 30.0, 'rainfall': 0.1, 'humidity': 45.0, 'pressure': 1020.0, 'today_raining': 0}
        },
        {
            'name': 'Mixed Conditions',
            'data': {'temperature': 26.0, 'rainfall': 2.1, 'humidity': 70.0, 'pressure': 1010.0, 'today_raining': 0}
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}:")
        print("-" * 30)
        data = test_case['data']
        print(f"Input: T={data['temperature']}°C, R={data['rainfall']}mm, "
              f"H={data['humidity']}%, P={data['pressure']}hPa, "
              f"Today Rain={'Yes' if data['today_raining'] else 'No'}")
        
        result = load_and_predict(model_filename, info_filename, data)
        
        if 'error' in result:
            print(f"❌ {result['error']}")
        else:
            print(f"🔮 {result['prediction']}")
            if 'confidence' in result:
                print(f"📊 Confidence: {result['confidence']}")
                print(f"🌧️ Rain Probability: {result['rain_probability']}")

test_saved_model()

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================

print(f"\n" + "=" * 70)
print("🎉 WEATHER PREDICTION MODEL TRAINING COMPLETE!")
print("=" * 70)
print(f"✅ Best Model: {best_model_name}")
print(f"✅ Test Accuracy: {best_accuracy:.1%}")
print(f"✅ Training Data: {len(df):,} samples")
print(f"✅ Features Used: {len(feature_columns)}")
print(f"")
print(f"📁 Files Created:")
print(f"   🤖 Model: {model_filename}")
print(f"   📋 Info: {info_filename}")
print(f"   📊 Plots: plots/ directory")
print(f"   📈 Data: data/ directory")
print(f"")
print(f"🚀 Ready for production use!")
print(f"   Use load_and_predict() function to make predictions")
print(f"   Model files are saved in the 'models/' directory")
print("=" * 70)

# Example usage for reference
print(f"\n📖 Example Usage:")
print(f"result = load_and_predict('{model_filename}', '{info_filename}', {{")
print(f"    'temperature': 28.5, 'rainfall': 3.2, 'humidity': 75,")
print(f"    'pressure': 1008, 'today_raining': 1")
print(f"}})")
print(f"print(result)")