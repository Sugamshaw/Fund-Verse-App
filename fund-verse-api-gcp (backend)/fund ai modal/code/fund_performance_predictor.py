"""
FUND VERSE - PREDICTIVE FUND PERFORMANCE MODEL
==============================================
Advanced ML System for NAV Prediction & Fund Performance Classification

Author: Fund Verse Analytics Team
Purpose: Google APM Portfolio Project
Features:
- NAV Prediction (Regression)
- Performance Classification (High/Medium/Low)
- Risk Assessment
- Smart Investment Recommendations

FIXED FOR WINDOWS COMPATIBILITY
"""

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             classification_report, confusion_matrix, accuracy_score)
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class FundPerformancePredictor:
    """
    Advanced ML system for fund performance prediction and analysis
    """
    
    def __init__(self):
        self.nav_model = None
        self.performance_classifier = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.feature_importance = {}
        
        # Create output directory if it doesn't exist
        self.output_dir = r'C:\Users\SUGAM SHAW\OneDrive\Desktop\fund ai\output'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"✅ Created output directory: {self.output_dir}")
        
    def load_and_merge_data(self, legal_entity_path, management_entity_path, 
                           fund_master_path, sub_fund_path, share_class_path):
        """Load and merge all data sources"""
        print("📊 Loading data from all sources...")
        
        # Load all datasets
        self.legal_entity = pd.read_csv(legal_entity_path)
        self.management_entity = pd.read_csv(management_entity_path)
        self.fund_master = pd.read_csv(fund_master_path)
        self.sub_fund = pd.read_csv(sub_fund_path)
        self.share_class = pd.read_csv(share_class_path)
        
        # Merge data for comprehensive analysis
        df = self.share_class.merge(self.fund_master, on='FUND_ID', how='left', suffixes=('_sc', '_fm'))
        df = df.merge(self.management_entity, on='MGMT_ID', how='left', suffixes=('', '_me'))
        # LE_ID from fund_master is already in df, use it to merge with legal_entity
        df = df.merge(self.legal_entity, on='LE_ID', how='left', suffixes=('', '_le'))
        
        print(f"✅ Loaded {len(df)} share class records with complete fund information")
        return df
    
    def exploratory_analysis(self, df):
        """Perform comprehensive exploratory data analysis"""
        print("\n" + "="*60)
        print("📈 EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        print("\n1. Dataset Overview:")
        print(f"   Total Records: {len(df)}")
        print(f"   Total Features: {len(df.columns)}")
        print(f"   Active Funds: {len(df[df['STATUS_fm'] == 'ACTIVE'])}")
        print(f"   Closed Funds: {len(df[df['STATUS_fm'] == 'CLOSED'])}")
        
        print("\n2. Key Financial Metrics:")
        print(f"   Average NAV: ${df['NAV'].mean():.2f}")
        print(f"   Total AUM: ${df['AUM'].sum():,.2f}")
        print(f"   Average AUM per Fund: ${df['AUM'].mean():,.2f}")
        print(f"   NAV Range: ${df['NAV'].min():.2f} - ${df['NAV'].max():.2f}")
        
        print("\n3. Fee Structure Analysis:")
        print(f"   Avg Management Fee: {df['FEE_MGMT'].mean()*100:.2f}%")
        print(f"   Avg Performance Fee: {df['PERF_FEE'].mean()*100:.2f}%")
        print(f"   Avg Total Expense Ratio: {df['EXPENSE_RATIO'].mean()*100:.2f}%")
        
        print("\n4. Fund Type Distribution:")
        print(df['FUND_TYPE'].value_counts())
        
        print("\n5. Geographic Distribution:")
        print(df['DOMICILE'].value_counts().head(10))
        
        return df
    
    def feature_engineering(self, df):
        """Create advanced features for ML models"""
        print("\n🔧 Engineering advanced features...")
        
        # Financial health indicators
        df['fee_ratio'] = df['FEE_MGMT'] / (df['EXPENSE_RATIO'] + 0.0001)
        df['performance_incentive'] = df['PERF_FEE'] * df['NAV']
        df['aum_per_nav'] = df['AUM'] / (df['NAV'] + 0.0001)
        df['total_fee_burden'] = df['FEE_MGMT'] + df['PERF_FEE'] + df['EXPENSE_RATIO']
        
        # Risk indicators
        df['high_expense_flag'] = (df['EXPENSE_RATIO'] > df['EXPENSE_RATIO'].quantile(0.75)).astype(int)
        df['high_perf_fee_flag'] = (df['PERF_FEE'] > df['PERF_FEE'].quantile(0.75)).astype(int)
        
        # Performance metrics
        df['nav_aum_efficiency'] = df['NAV'] * df['AUM'] / 1e10
        df['fee_efficiency'] = df['NAV'] / (df['total_fee_burden'] + 0.0001)
        
        # Status encoding
        df['is_active'] = (df['STATUS_fm'] == 'ACTIVE').astype(int)
        
        # Geographic diversity
        jurisdiction_counts = df['JURISDICTION'].value_counts()
        df['jurisdiction_popularity'] = df['JURISDICTION'].map(jurisdiction_counts)
        
        print(f"✅ Created {8} advanced financial features")
        return df
    
    def create_performance_labels(self, df):
        """Create performance classification labels"""
        print("\n🏷️  Creating performance classification labels...")
        
        # Calculate performance score based on multiple factors
        df['performance_score'] = (
            df['NAV'] * 0.3 +
            (df['AUM'] / 1e10) * 0.3 +
            (1 - df['EXPENSE_RATIO']) * 0.2 +
            df['fee_efficiency'] * 0.2
        )
        
        # Create performance categories
        perf_33 = df['performance_score'].quantile(0.33)
        perf_66 = df['performance_score'].quantile(0.66)
        
        def classify_performance(score):
            if score >= perf_66:
                return 'HIGH'
            elif score >= perf_33:
                return 'MEDIUM'
            else:
                return 'LOW'
        
        df['performance_class'] = df['performance_score'].apply(classify_performance)
        
        print(f"   Performance Distribution:")
        print(df['performance_class'].value_counts())
        
        return df
    
    def prepare_ml_features(self, df):
        """Prepare features for ML models"""
        print("\n🎯 Preparing ML features...")
        
        # Select numerical features for modeling
        feature_cols = [
            'FEE_MGMT', 'PERF_FEE', 'EXPENSE_RATIO', 'AUM',
            'fee_ratio', 'performance_incentive', 'aum_per_nav',
            'total_fee_burden', 'high_expense_flag', 'high_perf_fee_flag',
            'nav_aum_efficiency', 'fee_efficiency', 'is_active',
            'jurisdiction_popularity'
        ]
        
        # Clean data
        df_clean = df[feature_cols + ['NAV', 'performance_class']].dropna()
        
        X = df_clean[feature_cols]
        y_nav = df_clean['NAV']
        y_class = df_clean['performance_class']
        
        print(f"✅ Prepared {len(feature_cols)} features from {len(df_clean)} records")
        return X, y_nav, y_class, feature_cols
    
    def train_nav_predictor(self, X, y):
        """Train NAV prediction model"""
        print("\n" + "="*60)
        print("🤖 TRAINING NAV PREDICTION MODEL")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try multiple models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge Regression': Ridge(alpha=1.0),
        }
        
        best_model = None
        best_score = -np.inf
        results = {}
        
        print("\n📊 Comparing Models:")
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'predictions': y_pred
            }
            
            print(f"\n{name}:")
            print(f"   RMSE: ${rmse:.4f}")
            print(f"   MAE:  ${mae:.4f}")
            print(f"   R²:   {r2:.4f}")
            
            if r2 > best_score:
                best_score = r2
                best_model = model
                self.nav_model = model
        
        print(f"\n🏆 Best Model: {[k for k, v in models.items() if v == best_model][0]}")
        print(f"   R² Score: {best_score:.4f}")
        
        return results, X_test_scaled, y_test
    
    def train_performance_classifier(self, X, y):
        """Train performance classification model"""
        print("\n" + "="*60)
        print("🎯 TRAINING PERFORMANCE CLASSIFIER")
        print("="*60)
        
        # Encode labels
        y_encoded = self.le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest Classifier
        self.performance_classifier = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10
        )
        self.performance_classifier.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.performance_classifier.predict(X_test_scaled)
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n📊 Classification Results:")
        print(f"   Accuracy: {accuracy:.2%}")
        
        print("\n📋 Detailed Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.le.classes_
        ))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return cm, X_test_scaled, y_test, y_pred
    
    def analyze_feature_importance(self, X, feature_cols):
        """Analyze and visualize feature importance"""
        print("\n" + "="*60)
        print("🔍 FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        if hasattr(self.nav_model, 'feature_importances_'):
            importance = self.nav_model.feature_importances_
            self.feature_importance = dict(zip(feature_cols, importance))
            
            # Sort by importance
            sorted_features = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            print("\n🏆 Top 10 Most Important Features for NAV Prediction:")
            for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                print(f"   {i}. {feature}: {importance:.4f}")
            
            return sorted_features
        return None
    
    def generate_recommendations(self, df, top_n=10):
        """Generate smart investment recommendations"""
        print("\n" + "="*60)
        print("💡 SMART INVESTMENT RECOMMENDATIONS")
        print("="*60)
        
        # Calculate investment score
        df_rec = df.copy()
        df_rec['investment_score'] = (
            df_rec['NAV'] * 0.25 +
            (df_rec['AUM'] / 1e10) * 0.25 +
            (1 - df_rec['EXPENSE_RATIO']) * 0.25 +
            (1 - df_rec['total_fee_burden']) * 0.25
        )
        
        # Filter active funds
        active_funds = df_rec[df_rec['STATUS_fm'] == 'ACTIVE'].copy()
        
        # Top recommendations
        top_funds = active_funds.nlargest(top_n, 'investment_score')
        
        print(f"\n🌟 Top {top_n} Recommended Funds:\n")
        for idx, fund in top_funds.iterrows():
            print(f"{fund.name + 1}. {fund['FUND_NAME']}")
            print(f"   Fund ID: {fund['FUND_ID']}")
            print(f"   NAV: ${fund['NAV']:.2f}")
            print(f"   AUM: ${fund['AUM']:,.0f}")
            print(f"   Total Expense: {fund['EXPENSE_RATIO']*100:.2f}%")
            print(f"   Investment Score: {fund['investment_score']:.4f}")
            print()
        
        return top_funds
    
    def risk_assessment(self, df):
        """Comprehensive risk assessment"""
        print("\n" + "="*60)
        print("⚠️  RISK ASSESSMENT ANALYSIS")
        print("="*60)
        
        # Risk categories
        high_risk = df[
            (df['EXPENSE_RATIO'] > df['EXPENSE_RATIO'].quantile(0.75)) |
            (df['PERF_FEE'] > df['PERF_FEE'].quantile(0.75))
        ]
        
        low_risk = df[
            (df['EXPENSE_RATIO'] < df['EXPENSE_RATIO'].quantile(0.25)) &
            (df['PERF_FEE'] < df['PERF_FEE'].quantile(0.25))
        ]
        
        print(f"\n📊 Risk Distribution:")
        print(f"   High Risk Funds: {len(high_risk)} ({len(high_risk)/len(df)*100:.1f}%)")
        print(f"   Low Risk Funds: {len(low_risk)} ({len(low_risk)/len(df)*100:.1f}%)")
        print(f"   Medium Risk Funds: {len(df) - len(high_risk) - len(low_risk)}")
        
        print(f"\n⚠️  High Risk Characteristics:")
        print(f"   Avg Expense Ratio: {high_risk['EXPENSE_RATIO'].mean()*100:.2f}%")
        print(f"   Avg Performance Fee: {high_risk['PERF_FEE'].mean()*100:.2f}%")
        
        print(f"\n✅ Low Risk Characteristics:")
        print(f"   Avg Expense Ratio: {low_risk['EXPENSE_RATIO'].mean()*100:.2f}%")
        print(f"   Avg Performance Fee: {low_risk['PERF_FEE'].mean()*100:.2f}%")
        
        return high_risk, low_risk
    
    def create_visualizations(self, df, results, cm, feature_importance):
        """Create comprehensive visualizations"""
        print("\n📊 Creating visualizations...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. NAV Distribution
        ax1 = plt.subplot(3, 3, 1)
        df['NAV'].hist(bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('NAV ($)')
        plt.ylabel('Frequency')
        plt.title('NAV Distribution Across All Funds')
        plt.axvline(df['NAV'].mean(), color='red', linestyle='--', label='Mean')
        plt.legend()
        
        # 2. AUM vs NAV
        ax2 = plt.subplot(3, 3, 2)
        plt.scatter(df['NAV'], df['AUM']/1e9, alpha=0.5, c=df['EXPENSE_RATIO'], cmap='viridis')
        plt.xlabel('NAV ($)')
        plt.ylabel('AUM (Billions $)')
        plt.title('NAV vs AUM (colored by Expense Ratio)')
        plt.colorbar(label='Expense Ratio')
        
        # 3. Performance Distribution
        ax3 = plt.subplot(3, 3, 3)
        df['performance_class'].value_counts().plot(kind='bar', color=['green', 'yellow', 'red'])
        plt.xlabel('Performance Class')
        plt.ylabel('Count')
        plt.title('Fund Performance Distribution')
        plt.xticks(rotation=0)
        
        # 4. Fee Analysis
        ax4 = plt.subplot(3, 3, 4)
        fee_data = df[['FEE_MGMT', 'PERF_FEE', 'EXPENSE_RATIO']].mean() * 100
        fee_data.plot(kind='bar', color=['steelblue', 'orange', 'green'])
        plt.xlabel('Fee Type')
        plt.ylabel('Average Fee (%)')
        plt.title('Average Fee Structure Analysis')
        plt.xticks(rotation=45)
        
        # 5. Expense Ratio by Fund Type
        ax5 = plt.subplot(3, 3, 5)
        df.groupby('FUND_TYPE')['EXPENSE_RATIO'].mean().sort_values().plot(
            kind='barh', color='coral'
        )
        plt.xlabel('Average Expense Ratio')
        plt.title('Expense Ratio by Fund Type')
        
        # 6. Geographic AUM Distribution
        ax6 = plt.subplot(3, 3, 6)
        top_jurisdictions = df.groupby('DOMICILE')['AUM'].sum().nlargest(10) / 1e9
        top_jurisdictions.plot(kind='barh', color='teal')
        plt.xlabel('Total AUM (Billions $)')
        plt.title('Top 10 Jurisdictions by AUM')
        
        # 7. Feature Importance
        ax7 = plt.subplot(3, 3, 7)
        if feature_importance:
            top_10_features = dict(feature_importance[:10])
            plt.barh(list(top_10_features.keys()), list(top_10_features.values()), color='purple')
            plt.xlabel('Importance Score')
            plt.title('Top 10 Features for NAV Prediction')
        
        # 8. Confusion Matrix
        ax8 = plt.subplot(3, 3, 8)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['HIGH', 'LOW', 'MEDIUM'],
                   yticklabels=['HIGH', 'LOW', 'MEDIUM'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Performance Classification Confusion Matrix')
        
        # 9. Status Distribution
        ax9 = plt.subplot(3, 3, 9)
        df['STATUS_fm'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        plt.title('Fund Status Distribution')
        plt.ylabel('')
        
        plt.tight_layout()
        
        # Save to outputs directory (works on both Windows and Linux)
        output_path = os.path.join(self.output_dir, 'fund_analysis_dashboard.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved visualization dashboard to: {output_path}")
        
        return fig

def main():
    """Main execution function"""
    print("="*60)
    print("FUND VERSE - PREDICTIVE PERFORMANCE MODEL")
    print("Advanced ML System for Investment Intelligence")
    print("="*60)
    
    # Initialize predictor
    predictor = FundPerformancePredictor()
    
    # Define data paths - UPDATE THESE TO YOUR CSV LOCATIONS
    data_paths = {
        'legal_entity': r'C:\Users\SUGAM SHAW\OneDrive\Desktop\fund ai\input_database\legal_entity.csv',
        'management_entity': r'C:\Users\SUGAM SHAW\OneDrive\Desktop\fund ai\input_database\management_entity.csv',
        'fund_master': r'C:\Users\SUGAM SHAW\OneDrive\Desktop\fund ai\input_database\fund_master.csv',
        'sub_fund': r'C:\Users\SUGAM SHAW\OneDrive\Desktop\fund ai\input_database\sub_fund.csv',
        'share_class': r'C:\Users\SUGAM SHAW\OneDrive\Desktop\fund ai\input_database\share_class.csv',
    }
    
    # Check if files exist
    print("\n🔍 Checking for CSV files...")
    for name, path in data_paths.items():
        if os.path.exists(path):
            print(f"   ✅ Found: {path}")
        else:
            print(f"   ❌ Missing: {path}")
            print(f"\n⚠️  ERROR: Please place {path} in the same directory as this script!")
            print(f"   Current directory: {os.getcwd()}")
            return None, None, None
    
    # Load data
    df = predictor.load_and_merge_data(
        data_paths['legal_entity'],
        data_paths['management_entity'],
        data_paths['fund_master'],
        data_paths['sub_fund'],
        data_paths['share_class']
    )
    
    # Exploratory analysis
    df = predictor.exploratory_analysis(df)
    
    # Feature engineering
    df = predictor.feature_engineering(df)
    
    # Create performance labels
    df = predictor.create_performance_labels(df)
    
    # Prepare ML features
    X, y_nav, y_class, feature_cols = predictor.prepare_ml_features(df)
    
    # Train NAV predictor
    results, X_test, y_test = predictor.train_nav_predictor(X, y_nav)
    
    # Train performance classifier
    cm, X_test_class, y_test_class, y_pred_class = predictor.train_performance_classifier(X, y_class)
    
    # Feature importance
    feature_importance = predictor.analyze_feature_importance(X, feature_cols)
    
    # Risk assessment
    high_risk, low_risk = predictor.risk_assessment(df)
    
    # Generate recommendations
    top_funds = predictor.generate_recommendations(df)
    
    # Create visualizations
    predictor.create_visualizations(df, results, cm, feature_importance)
    
    # Save processed data to outputs directory
    csv_output_path = os.path.join(predictor.output_dir, 'processed_fund_data.csv')
    df.to_csv(csv_output_path, index=False)
    print(f"\n✅ Saved processed data to: {csv_output_path}")
    
    print("\n" + "="*60)
    print("✨ ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    print(f"1. {os.path.join(predictor.output_dir, 'fund_analysis_dashboard.png')} - Comprehensive visualizations")
    print(f"2. {csv_output_path} - Enhanced dataset with predictions")
    
    return predictor, df, results

if __name__ == "__main__":
    predictor, df, results = main()