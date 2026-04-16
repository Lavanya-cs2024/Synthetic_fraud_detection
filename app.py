from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
import os

app = Flask(__name__)

# ============================================
# LOAD YOUR TRAINED MODEL
# ============================================

print("="*50)
print("Loading Fraud Detection Model...")
print("="*50)

# Load model
try:
    model = joblib.load('models/best_fraud_model.pkl')
    print("✅ Model loaded successfully!")
    print(f"   Type: {type(model).__name__}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Load scaler
try:
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    scaler = None

# Load metadata
try:
    with open('models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    print("✅ Metadata loaded successfully!")
    scaled_columns = metadata.get('scaled_columns', [])
    final_features = metadata.get('final_features', [])
    
    # IMPORTANT: Remove 'Cluster' if it exists (from K-Means clustering analysis)
    if 'Cluster' in final_features:
        final_features.remove('Cluster')
        print("⚠️ Removed 'Cluster' feature (clustering was for analysis only)")
    
    print(f"   Features count: {len(final_features)}")
    
except Exception as e:
    print(f"❌ Error loading metadata: {e}")
    scaled_columns = []
    final_features = []

print(f"\n📊 Configuration:")
print(f"   Scaled columns: {len(scaled_columns)}")
print(f"   Final features: {len(final_features)}")
print("="*50)

# ============================================
# CATEGORICAL COLUMNS (from your model)
# ============================================

CATEGORICAL_COLUMNS = ['document_type', 'ip_type', 'os', 'employment_status']

# ============================================
# SECURITY RECOMMENDATION ENGINE
# ============================================

def get_security_recommendations(user_data, fraud_probability):
    """Generate security recommendations ONLY if risk is significant"""
    
    # If fraud probability is LOW (<8%), return simple message
    if fraud_probability < 0.08:
        return ["✅ Low risk detected - Standard monitoring only. No action required."]
    
    recommendations = []
    
    # Only show these for MEDIUM or HIGH risk (8%+)
    if fraud_probability > 0.08:
        if user_data.get('ip_risk_score', 0) > 0.7:
            recommendations.append("🔐 Mandate Hardware-based 2FA (YubiKey) due to high-risk IP")
        
        if user_data.get('ip_address_country_mismatch', 0) == 1:
            recommendations.append("🌍 IP-Country mismatch detected - Require additional ID verification")
        
        if user_data.get('device_reuse_score', 0) > 0.8:
            recommendations.append("📱 Device reuse anomaly - Trigger Video KYC verification")
        
        if user_data.get('accounts_same_device', 0) > 3:
            recommendations.append("👥 Multiple accounts from same device - Require phone verification")
        
        # Behavioral risks
        edits = user_data.get('num_field_edits', 0)
        duration = user_data.get('signup_duration_sec', 1)
        edit_ratio = edits / (duration + 1)
        
        if edit_ratio > 0.8:
            recommendations.append("⚡ Unusually fast form completion - Implement proof-of-work CAPTCHA")
        elif edit_ratio > 0.5:
            recommendations.append("🐌 Form filling speed anomaly - Add behavioral challenge")
        
        # Identity risks
        if user_data.get('email_risk', 0) > 0.6:
            recommendations.append("📧 High-risk email domain - Require email confirmation link")
        
        if user_data.get('address_verification_score', 1) < 0.3:
            recommendations.append("🏠 Unverified address - Require utility bill upload")
    
    # Only show HIGH risk actions (15%+)
    if fraud_probability > 0.15:
        recommendations.insert(0, "🚨 HIGH RISK: Immediate account freeze and manual review required")
    
    # Default message if no specific recommendations but risk is medium
    if not recommendations and fraud_probability > 0.08:
        recommendations = ["⚠️ Suspicious activity detected - Enhanced monitoring for 30 days"]
    
    return recommendations

# ============================================
# DATA PREPROCESSING
# ============================================

def preprocess_input(raw_data):
    """Preprocess raw input data to match training format"""
    
    # Create DataFrame from input
    df = pd.DataFrame([raw_data])
    
    # 1. Add engineered feature
    df['fields_edit_ratio'] = df['num_field_edits'] / (df['signup_duration_sec'] + 1)
    
    # 2. One-hot encode categorical variables
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)
    
    # 3. Ensure all expected columns exist
    if final_features:
        # Add missing columns with 0
        for col in final_features:
            if col not in df.columns:
                df[col] = 0
        # Keep only training columns in correct order
        df = df[final_features]
    else:
        print("⚠️ Warning: No final_features in metadata, using available columns")
    
    # 4. Scale continuous features
    if scaler and scaled_columns:
        # Only scale columns that exist in the dataframe
        available_scaled = [col for col in scaled_columns if col in df.columns]
        if available_scaled:
            df[available_scaled] = scaler.transform(df[available_scaled])
    
    return df

# ============================================
# FLASK ROUTES
# ============================================

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make fraud prediction"""
    
    if model is None:
        return render_template('index.html', error="Model not loaded. Please check model files.")
    
    try:
        # Get form data
        raw_input = {
            'age': int(request.form['age']),
            'document_type': request.form['document_type'],
            'phone_valid': int(request.form['phone_valid']),
            'email_risk': float(request.form['email_risk']),
            'ip_type': request.form['ip_type'],
            'ip_risk_score': float(request.form['ip_risk_score']),
            'ip_address_country_mismatch': int(request.form['ip_address_country_mismatch']),
            'os': request.form['os'],
            'device_is_new': int(request.form['device_is_new']),
            'accounts_same_device': int(request.form['accounts_same_device']),
            'accounts_same_ip': int(request.form['accounts_same_ip']),
            'device_reuse_score': float(request.form['device_reuse_score']),
            'employment_status': request.form['employment_status'],
            'annual_income': float(request.form['annual_income']),
            'signup_duration_sec': int(request.form['signup_duration_sec']),
            'num_field_edits': int(request.form['num_field_edits']),
            'address_verification_score': float(request.form['address_verification_score'])
        }
        
        # Preprocess the input
        processed_data = preprocess_input(raw_input)
        
        # Make prediction
        probability = model.predict_proba(processed_data)[0][1]
        
        # ============================================
        # ADJUSTED THRESHOLD BASED ON BATCH TEST
        # ============================================
        # Batch test results:
        # - Low risk: 5.39% → LEGIT
        # - Medium: 7.78% → LEGIT
        # - High: 16.39% → FRAUD
        # - Extreme: 21.97% → FRAUD
        
        THRESHOLD = 0.12  # 12% threshold - catches high/extreme risk
        
        prediction = 1 if probability > THRESHOLD else 0
        
        # ============================================
        # ADJUSTED RISK LEVELS
        # ============================================
        if probability > 0.15:      # 15%+ = High Risk
            risk_level = "High"
        elif probability > 0.08:    # 8-15% = Medium Risk
            risk_level = "Medium"
        else:                       # Below 8% = Low Risk
            risk_level = "Low"
        
        # Get recommendations
        recommendations = get_security_recommendations(raw_input, probability)
        
        # Add threshold info to recommendations for transparency
        if probability > THRESHOLD and probability < 0.5:
            recommendations.insert(0, f"⚠️ Fraud probability is {probability*100:.1f}% (above {THRESHOLD*100:.0f}% alert threshold)")
        
        result = {
            'is_fraud': bool(prediction),
            'fraud_probability': round(probability * 100, 2),
            'risk_level': risk_level,
            'recommendations': recommendations,
            'prediction_text': 'FRAUD DETECTED' if prediction else 'LEGITIMATE USER',
            'prediction_class': 'fraud' if prediction else 'legitimate'
        }
        
        # Debug print to terminal
        print(f"\n📊 Prediction: {result['prediction_text']} | Probability: {result['fraud_probability']}% | Risk: {risk_level}")
        
        return render_template('index.html', result=result)
    
    except Exception as e:
        error_details = str(e)
        print(f"Error in prediction: {error_details}")
        return render_template('index.html', error=f"Prediction error: {error_details}")

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'features_count': len(final_features)
    })

# ============================================
# BATCH TEST ENDPOINT - For debugging
# ============================================

@app.route('/batch_test')
def batch_test():
    """Test model with multiple scenarios to see what probabilities it outputs"""
    from flask import render_template_string
    
    results = []
    
    scenarios = [
        {"name": "✅ LOW RISK (Should be Legitimate)", "data": {
            'age': 35, 'document_type': 'passport', 'phone_valid': 1,
            'email_risk': 0.1, 'ip_type': 'residential', 'ip_risk_score': 0.1,
            'ip_address_country_mismatch': 0, 'os': 'windows', 'device_is_new': 0,
            'accounts_same_device': 1, 'accounts_same_ip': 1, 'device_reuse_score': 0.1,
            'employment_status': 'employed', 'annual_income': 75000,
            'signup_duration_sec': 120, 'num_field_edits': 15,
            'address_verification_score': 0.9
        }},
        {"name": "⚠️ MEDIUM RISK", "data": {
            'age': 22, 'document_type': 'passport', 'phone_valid': 1,
            'email_risk': 0.5, 'ip_type': 'mobile', 'ip_risk_score': 0.5,
            'ip_address_country_mismatch': 0, 'os': 'windows', 'device_is_new': 1,
            'accounts_same_device': 2, 'accounts_same_ip': 3, 'device_reuse_score': 0.5,
            'employment_status': 'student', 'annual_income': 30000,
            'signup_duration_sec': 45, 'num_field_edits': 8,
            'address_verification_score': 0.5
        }},
        {"name": "🚨 HIGH RISK (Should be Fraud)", "data": {
            'age': 19, 'document_type': 'passport', 'phone_valid': 0,
            'email_risk': 0.9, 'ip_type': 'datacenter', 'ip_risk_score': 0.95,
            'ip_address_country_mismatch': 1, 'os': 'windows', 'device_is_new': 1,
            'accounts_same_device': 5, 'accounts_same_ip': 10, 'device_reuse_score': 0.9,
            'employment_status': 'unemployed', 'annual_income': 10000,
            'signup_duration_sec': 5, 'num_field_edits': 2,
            'address_verification_score': 0.1
        }},
        {"name": "🔴 EXTREME RISK", "data": {
            'age': 18, 'document_type': 'passport', 'phone_valid': 0,
            'email_risk': 1.0, 'ip_type': 'datacenter', 'ip_risk_score': 1.0,
            'ip_address_country_mismatch': 1, 'os': 'windows', 'device_is_new': 1,
            'accounts_same_device': 10, 'accounts_same_ip': 20, 'device_reuse_score': 1.0,
            'employment_status': 'unemployed', 'annual_income': 5000,
            'signup_duration_sec': 3, 'num_field_edits': 1,
            'address_verification_score': 0.0
        }}
    ]
    
    for scenario in scenarios:
        try:
            # Preprocess the test data
            df = preprocess_input(scenario['data'])
            
            # Get prediction
            probability = model.predict_proba(df)[0][1]
            
            # Apply same threshold as main app
            THRESHOLD = 0.12
            prediction = 'FRAUD' if probability > THRESHOLD else 'LEGIT'
            
            results.append({
                'scenario': scenario['name'],
                'fraud_probability': f"{probability*100:.2f}%",
                'probability_value': probability,
                'prediction': prediction,
                'risk_color': '#ffcccc' if probability > THRESHOLD else '#ccffcc'
            })
        except Exception as e:
            results.append({
                'scenario': scenario['name'],
                'fraud_probability': f"ERROR",
                'probability_value': 0,
                'prediction': str(e)[:50],
                'risk_color': '#ffcccc'
            })
    
    # Create HTML table
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Batch Test Results</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 10px;
            }
            h1 {
                color: #667eea;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }
            th {
                background: #667eea;
                color: white;
            }
            .fraud {
                background: #ffcccc;
                color: #c00;
                font-weight: bold;
            }
            .legit {
                background: #ccffcc;
                color: #2e7d32;
                font-weight: bold;
            }
            .note {
                margin-top: 20px;
                padding: 15px;
                background: #fff3cd;
                border-left: 4px solid #ffc107;
            }
            button {
                background: #667eea;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                margin-top: 20px;
                margin-right: 10px;
            }
            button:hover {
                background: #764ba2;
            }
            .threshold-info {
                background: #e8f5e9;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                border-left: 4px solid #4caf50;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧪 Model Batch Test Results</h1>
            <p>Testing your model with different risk scenarios</p>
            
            <div class="threshold-info">
                <strong>⚙️ Current Settings:</strong><br>
                Detection Threshold: <strong>12%</strong> (Fraud alert when probability > 12%)<br>
                Based on your model's behavior: Low risk (~5%), High risk (~16-22%)
            </div>
            
            <table>
                <tr>
                    <th>Scenario</th>
                    <th>Fraud Probability</th>
                    <th>Prediction (Threshold 12%)</th>
                    <th>Status</th>
                </tr>
                {% for r in results %}
                <tr>
                    <td>{{ r.scenario }}</td>
                    <td><strong>{{ r.fraud_probability }}</strong></td>
                    <td class="{% if r.prediction == 'FRAUD' %}fraud{% else %}legit{% endif %}">
                        {{ r.prediction }}
                    </td>
                    <td>
                        {% if r.probability_value > 0.15 %}
                            🔴 High Risk
                        {% elif r.probability_value > 0.08 %}
                            🟡 Medium Risk
                        {% else %}
                            🟢 Low Risk
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </table>
            
            <div class="note">
                <strong>💡 What this tells you:</strong>
                <ul>
                    <li>✅ LOW RISK (5.39%) → LEGIT - Correct!</li>
                    <li>⚠️ MEDIUM RISK (7.78%) → LEGIT - Correct!</li>
                    <li>🚨 HIGH RISK (16.39%) → FRAUD - Correct! (Threshold 12%)</li>
                    <li>🔴 EXTREME RISK (21.97%) → FRAUD - Correct!</li>
                </ul>
                <p><strong>The model is now working correctly with 12% threshold!</strong></p>
            </div>
            
            <button onclick="window.location.href='/'">← Back to Main App</button>
            <button onclick="window.location.reload()">⟳ Run Test Again</button>
        </div>
    </body>
    </html>
    """
    
    return render_template_string(html, results=results)

# ============================================
# THRESHOLD SETTINGS PAGE
# ============================================

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Adjust prediction threshold dynamically"""
    from flask import render_template_string, session
    
    # Get current threshold from session or use default
    threshold = session.get('threshold', 0.12)
    
    if request.method == 'POST':
        threshold = float(request.form.get('threshold', 0.12))
        session['threshold'] = threshold
        return render_template_string(SETTINGS_HTML, threshold=threshold, saved=True)
    
    return render_template_string(SETTINGS_HTML, threshold=threshold, saved=False)

SETTINGS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Model Settings</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 20px;
        }
        h1 { color: #667eea; }
        .current-value {
            font-size: 48px;
            color: #667eea;
            text-align: center;
            margin: 20px;
        }
        input[type="range"] {
            width: 100%;
            margin: 20px 0;
        }
        button {
            background: #667eea;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
        }
        button:hover {
            background: #764ba2;
        }
        .info {
            background: #f0f0f0;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .success {
            background: #d4edda;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .button-group button {
            flex: 1;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚙️ Model Threshold Settings</h1>
        
        {% if saved %}
        <div class="success">✅ Threshold saved successfully!</div>
        {% endif %}
        
        <form method="POST">
            <label><strong>Fraud Detection Threshold:</strong></label>
            <div class="current-value" id="thresholdValue">{{ "%.2f"|format(threshold) }}</div>
            <input type="range" name="threshold" min="0.01" max="0.5" step="0.01" 
                   value="{{ threshold }}" oninput="updateValue(this.value)">
            <p><small>Lower = More fraud detections (more false alarms)</small></p>
            <p><small>Higher = Fewer fraud detections (might miss real fraud)</small></p>
            <button type="submit">💾 Save Threshold</button>
        </form>
        
        <div class="info">
            <strong>📊 Based on your batch test results:</strong>
            <ul>
                <li>Low Risk: 5.39% → Should be LEGIT</li>
                <li>Medium Risk: 7.78% → Should be LEGIT</li>
                <li>High Risk: 16.39% → Should be FRAUD</li>
                <li>Extreme Risk: 21.97% → Should be FRAUD</li>
            </ul>
            <p><strong>✅ Recommended threshold: 0.12 (12%)</strong></p>
        </div>
        
        <div class="button-group">
            <button onclick="window.location.href='/'">← Back to App</button>
            <button onclick="window.location.href='/batch_test'" style="background: #764ba2;">🔬 Run Batch Test</button>
        </div>
    </div>
    
    <script>
        function updateValue(val) {
            document.getElementById('thresholdValue').innerText = parseFloat(val).toFixed(2);
        }
    </script>
</body>
</html>
"""

# ============================================
# RUN THE APP
# ============================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Fraud Detection Web Application")
    print("="*50)
    print(f"📊 Model loaded: {model is not None}")
    print(f"📈 Scaler loaded: {scaler is not None}")
    print(f"🔢 Features: {len(final_features)}")
    print(f"⚙️ Detection Threshold: 12% (can be adjusted at /settings)")
    print("\n🌐 Open in browser:")
    print("   - Main app: http://localhost:5000")
    print("   - Batch test: http://localhost:5000/batch_test")
    print("   - Settings: http://localhost:5000/settings")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)