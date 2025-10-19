"""
FUND VERSE - ML MODEL TRAINING SCRIPT
======================================
Train and save ML models for NAV prediction and performance classification

This script will create 4 files in the models/ directory:
1. nav_model.pkl - NAV prediction model
2. performance_classifier.pkl - Performance classification model
3. scaler.pkl - Feature scaler
4. label_encoder.pkl - Label encoder

Author: Fund Verse Team
"""

import mysql.connector
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def get_data_from_database():
    """Fetch data from MySQL database"""
    print("\n📊 Connecting to database...")
    
    # Database configuration
    db_config = {
        'host': '35.232.60.43',
        'user': 'fund_db',
        'password': 'Password123',
        'database': 'fund_system',
        'port': 3306
    }
    
    try:
        # Connect to database
        conn = mysql.connector.connect(**db_config)
        print("✅ Database connected successfully")
        
        # SQL query to fetch all required data
        query = """
            SELECT 
                fm.FUND_ID,
                fm.FUND_NAME,
                fm.STATUS as FUND_STATUS,
                sc.SC_ID,
                sc.NAV,
                sc.AUM,
                sc.FEE_MGMT,
                sc.PERF_FEE,
                sc.EXPENSE_RATIO,
                sc.CURRENCY,
                sc.DISTRIBUTION,
                le.JURISDICTION,
                le.DOMICILE
            FROM fund_master fm
            LEFT JOIN share_class sc ON fm.FUND_ID = sc.FUND_ID
            LEFT JOIN legal_entity le ON fm.LE_ID = le.LE_ID
            WHERE sc.NAV IS NOT NULL 
            AND sc.AUM IS NOT NULL
            AND sc.FEE_MGMT IS NOT NULL
        """
        
        # Read data into DataFrame
        df = pd.read_sql(query, conn)
        conn.close()
        
        print(f"✅ Fetched {len(df)} records from database")
        return df
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        print("\n⚠️  Using sample data instead...")
        return create_sample_data()


def create_sample_data():
    """Create sample data if database connection fails"""
    print("📝 Creating sample dataset...")
    
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'FUND_ID': [f'F{str(i).zfill(6)}' for i in range(n_samples)],
        'FUND_NAME': [f'Fund {i}' for i in range(n_samples)],
        'FUND_STATUS': np.random.choice(['ACTIVE', 'INACTIVE'], n_samples, p=[0.8, 0.2]),
        'SC_ID': [f'SC{str(i).zfill(6)}' for i in range(n_samples)],
        'NAV': np.random.uniform(10, 500, n_samples),
        'AUM': np.random.uniform(1e8, 1e12, n_samples),
        'FEE_MGMT': np.random.uniform(0.01, 0.05, n_samples),
        'PERF_FEE': np.random.uniform(0.05, 0.25, n_samples),
        'EXPENSE_RATIO': np.random.uniform(0.02, 0.50, n_samples),
        'CURRENCY': np.random.choice(['USD', 'EUR', 'GBP'], n_samples),
        'DISTRIBUTION': np.random.choice(['ACCUMULATION', 'DISTRIBUTION'], n_samples),
        'JURISDICTION': np.random.choice(['Luxembourg', 'Ireland', 'USA'], n_samples),
        'DOMICILE': np.random.choice(['Luxembourg', 'Ireland', 'USA'], n_samples)
    }
    
    df = pd.DataFrame(data)
    print(f"✅ Created {len(df)} sample records")
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """Create advanced features for ML models"""
    print("\n🔧 Engineering features...")
    
    # Create copy to avoid modifying original
    df = df.copy()
    
    # 1. Fee Ratios
    df['fee_ratio'] = df['FEE_MGMT'] / (df['EXPENSE_RATIO'] + 0.0001)
    
    # 2. Performance Incentive
    df['performance_incentive'] = df['PERF_FEE'] * df['NAV']
    
    # 3. AUM per NAV
    df['aum_per_nav'] = df['AUM'] / (df['NAV'] + 0.0001)
    
    # 4. Total Fee Burden
    df['total_fee_burden'] = df['FEE_MGMT'] + df['PERF_FEE'] + df['EXPENSE_RATIO']
    
    # 5. High Expense Flag
    df['high_expense_flag'] = (df['EXPENSE_RATIO'] > 0.20).astype(int)
    
    # 6. High Performance Fee Flag
    df['high_perf_fee_flag'] = (df['PERF_FEE'] > 0.15).astype(int)
    
    # 7. NAV-AUM Efficiency
    df['nav_aum_efficiency'] = df['NAV'] * df['AUM'] / 1e10
    
    # 8. Fee Efficiency
    df['fee_efficiency'] = df['NAV'] / (df['FEE_MGMT'] + df['PERF_FEE'] + df['EXPENSE_RATIO'] + 0.0001)
    
    # 9. Active Status
    df['is_active'] = (df['FUND_STATUS'] == 'ACTIVE').astype(int)
    
    # 10. Jurisdiction Popularity (simplified)
    jurisdiction_counts = df['JURISDICTION'].value_counts()
    df['jurisdiction_popularity'] = df['JURISDICTION'].map(jurisdiction_counts) / len(df)
    
    # 11. Performance Classification (target for classification)
    # Based on NAV and fee efficiency
    performance_score = (
        df['NAV'] / df['NAV'].max() * 0.4 +
        df['fee_efficiency'] / df['fee_efficiency'].max() * 0.3 +
        df['AUM'] / df['AUM'].max() * 0.3
    )
    
    df['performance_class'] = pd.cut(
        performance_score,
        bins=3,
        labels=['LOW', 'MEDIUM', 'HIGH']
    )
    
    print(f"✅ Created {len(df.columns)} features")
    return df


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_nav_prediction_model(df):
    """Train NAV prediction model"""
    print("\n🤖 Training NAV Prediction Model...")
    
    # Select features
    feature_cols = [
        'FEE_MGMT', 'PERF_FEE', 'EXPENSE_RATIO', 'AUM',
        'fee_ratio', 'performance_incentive', 'aum_per_nav',
        'total_fee_burden', 'high_expense_flag', 'high_perf_fee_flag',
        'nav_aum_efficiency', 'fee_efficiency', 'is_active',
        'jurisdiction_popularity'
    ]
    
    X = df[feature_cols].copy()
    y = df['NAV'].copy()
    
    # Handle any missing values
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Ridge Regression model
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"✅ NAV Prediction Model Trained")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAE: ${mae:.2f}")
    
    return model, scaler, feature_cols


def train_performance_classifier(df):
    """Train performance classification model"""
    print("\n🎯 Training Performance Classification Model...")
    
    # Select features
    feature_cols = [
        'FEE_MGMT', 'PERF_FEE', 'EXPENSE_RATIO', 'AUM',
        'fee_ratio', 'performance_incentive', 'aum_per_nav',
        'total_fee_burden', 'high_expense_flag', 'high_perf_fee_flag',
        'nav_aum_efficiency', 'fee_efficiency', 'is_active',
        'jurisdiction_popularity'
    ]
    
    X = df[feature_cols].copy()
    y = df['performance_class'].copy()
    
    # Handle any missing values
    X = X.fillna(X.mean())
    
    # Remove any NaN in target
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Note: We'll reuse the scaler from NAV prediction
    # But for classification, we could also train a separate one
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Performance Classification Model Trained")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    return model, label_encoder


# ============================================================================
# MODEL SAVING
# ============================================================================

def save_models(nav_model, performance_model, scaler, label_encoder):
    """Save all trained models"""
    print("\n💾 Saving models...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save models
    pickle.dump(nav_model, open('models/nav_model.pkl', 'wb'))
    pickle.dump(performance_model, open('models/performance_classifier.pkl', 'wb'))
    pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
    pickle.dump(label_encoder, open('models/label_encoder.pkl', 'wb'))
    
    print("✅ Models saved successfully:")
    print("   📁 models/nav_model.pkl")
    print("   📁 models/performance_classifier.pkl")
    print("   📁 models/scaler.pkl")
    print("   📁 models/label_encoder.pkl")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("🤖 FUND VERSE ML MODEL TRAINING")
    print("="*60)
    
    # Step 1: Get data
    df = get_data_from_database()
    
    # Step 2: Engineer features
    df = engineer_features(df)
    
    # Step 3: Train NAV prediction model
    nav_model, scaler, feature_cols = train_nav_prediction_model(df)
    
    # Step 4: Train performance classifier
    performance_model, label_encoder = train_performance_classifier(df)
    
    # Step 5: Save models
    save_models(nav_model, performance_model, scaler, label_encoder)
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print("\n✅ You can now run the API server:")
    print("   python ml_api_server_FIXED.py")
    print("\n")
    
    return df, nav_model, performance_model


if __name__ == "__main__":
    df, nav_model, performance_model = main()