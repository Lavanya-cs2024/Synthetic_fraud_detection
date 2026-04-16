# check_files.py
import os
import sys

print("="*50)
print("Checking Your Project Setup")
print("="*50)

# Check current directory
print(f"\n📁 Current directory: {os.getcwd()}")

# Check if files exist
files_to_check = [
    'app.py',
    'requirements.txt',
    'models/best_fraud_model.pkl',
    'models/scaler.pkl',
    'models/model_metadata.json',
    'templates/index.html',
    'static/style.css'
]

print("\n📋 File Check:")
for file in files_to_check:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"   ✅ {file} ({size} bytes)")
    else:
        print(f"   ❌ {file} - MISSING!")

# Check Python version
print(f"\n🐍 Python version: {sys.version}")

# Try importing Flask
try:
    import flask
    print(f"✅ Flask installed (version {flask.__version__})")
except:
    print("❌ Flask NOT installed! Run: pip install flask")

# Try importing joblib
try:
    import joblib
    print("✅ joblib installed")
except:
    print("❌ joblib NOT installed! Run: pip install joblib")

print("\n" + "="*50)