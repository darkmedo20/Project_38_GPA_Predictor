"""
Simple Model Comparison - Choose Best Performer
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load or create data
print("📊 Loading data...")
try:
    df = pd.read_csv('data/cleaned_student_performance.csv')
    X = df.drop(['StudentID', 'Final_Year_GPA'], axis=1, errors='ignore')
    y = df['Final_Year_GPA']
    print(f"   Real data: {X.shape[0]} samples, {X.shape[1]} features")
except:
    print("   Using synthetic data")
    np.random.seed(42)
    X = pd.DataFrame({
        'Year1_GPA': np.random.uniform(2.0, 4.0, 100),
        'Year2_GPA': np.random.uniform(2.0, 4.0, 100),
        'Year3_GPA': np.random.uniform(2.0, 4.0, 100),
        'Credit_Hours': np.random.uniform(12, 20, 100),
    })
    y = 0.4*X['Year3_GPA'] + 0.3*X['Year2_GPA'] + 0.2*X['Year1_GPA'] + np.random.normal(0, 0.2, 100)
    y = pd.Series(np.clip(y, 0, 4), name='Final_Year_GPA')

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate
results = []
print("\n🎯 Training models...")
for name, model in models.items():
    print(f"   {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append({'Model': name, 'R² Score': r2, 'RMSE': rmse})

# Create results table
results_df = pd.DataFrame(results).sort_values('R² Score', ascending=False)
print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)
print(results_df.to_string(index=False))

# Select best model
best_model_name = results_df.iloc[0]['Model']
best_r2 = results_df.iloc[0]['R² Score']
best_rmse = results_df.iloc[0]['RMSE']

print("\n" + "="*60)
print(f"🏆 BEST PERFORMER: {best_model_name}")
print(f"   R² Score: {best_r2:.4f}")
print(f"   RMSE: {best_rmse:.4f} GPA points")
print("="*60)

# Visual comparison
plt.figure(figsize=(10, 6))
models_list = results_df['Model']
r2_scores = results_df['R² Score']

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = plt.bar(models_list, r2_scores, color=colors, edgecolor='black')

# Highlight best model
bars[0].set_color('#2E7D32')
bars[0].set_edgecolor('black')

plt.xlabel('Model', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.title('Model Performance Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)

# Add value labels
for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('best_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ Comparison chart saved: best_model_comparison.png")
print("✅ Best model selected and ready for deployment!")