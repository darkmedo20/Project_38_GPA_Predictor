"""
Script to generate all visualizations for the project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_dataset
from model_pipeline import GPAModelPipeline

def create_all_visualizations():
    # Load data
    X, y, features = load_dataset('data/cleaned_student_performance.csv')
    
    # Train model
    pipeline = GPAModelPipeline(model_type='random_forest')
    pipeline.train_model(X, y)
    
    # Generate basic visualizations
    from model_pipeline import generate_visualizations  # Add the function above
    generate_visualizations(pipeline, X, y, X, y)
    
    # Create additional visualizations
    
    # 1. GPA Distribution Over Years
    plt.figure(figsize=(12, 8))
    years = ['Year1_GPA', 'Year2_GPA', 'Year3_GPA', 'Final_Year_GPA']
    data = [X['Year1_GPA'], X['Year2_GPA'], X['Year3_GPA'], y]
    
    plt.subplot(2, 2, 1)
    plt.hist(X['Year1_GPA'], bins=15, edgecolor='black', alpha=0.7)
    plt.title('Year 1 GPA Distribution')
    plt.xlabel('GPA')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 2)
    plt.hist(X['Year2_GPA'], bins=15, edgecolor='black', alpha=0.7)
    plt.title('Year 2 GPA Distribution')
    plt.xlabel('GPA')
    
    plt.subplot(2, 2, 3)
    plt.hist(X['Year3_GPA'], bins=15, edgecolor='black', alpha=0.7)
    plt.title('Year 3 GPA Distribution')
    plt.xlabel('GPA')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 4)
    plt.hist(y, bins=15, edgecolor='black', alpha=0.7)
    plt.title('Final Year GPA Distribution')
    plt.xlabel('GPA')
    
    plt.tight_layout()
    plt.savefig('visualizations/gpa_distributions.png', dpi=300)
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    correlation_data = pd.concat([X, y.rename('Final_Year_GPA')], axis=1)
    correlation_matrix = correlation_data.corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=.5, 
                cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png', dpi=300)
    plt.close()
    
    # 3. Credit Hours vs GPA Scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(X['Credit_Hours_Avg'], y, alpha=0.6, c='blue', edgecolors='black')
    plt.xlabel('Average Credit Hours')
    plt.ylabel('Final Year GPA')
    plt.title('Credit Hours vs Final GPA')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(X['Credit_Hours_Avg'], y, 1)
    p = np.poly1d(z)
    plt.plot(X['Credit_Hours_Avg'], p(X['Credit_Hours_Avg']), "r--", alpha=0.8)
    
    plt.savefig('visualizations/credit_hours_vs_gpa.png', dpi=300)
    plt.close()
    
    print("\n✅ All visualizations generated successfully!")
    print("📁 Check the 'visualizations/' folder for:")
    print("   1. actual_vs_predicted.png")
    print("   2. feature_importance.png")
    print("   3. error_distribution.png")
    print("   4. gpa_distributions.png")
    print("   5. correlation_heatmap.png")
    print("   6. credit_hours_vs_gpa.png")

if __name__ == "__main__":
    create_all_visualizations()