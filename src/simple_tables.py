"""
Ultra Simple Visual Tables
Creates 3 essential table images quickly
"""

import pandas as pd
import matplotlib.pyplot as plt

# 1. Performance Summary Table
performance = pd.DataFrame({
    'Metric': ['R² Score', 'RMSE', 'Accuracy', 'Precision', 'Recall'],
    'Value': ['0.823', '0.278', '86.0%', '86.4%', '89.5%'],
    'Interpretation': ['Excellent', 'Good', 'Good', 'Good', 'Excellent']
})

# 2. Top Features Table
features = pd.DataFrame({
    'Rank': [1, 2, 3, 4, 5],
    'Feature': ['Year 3 GPA', 'Year 2 GPA', 'Year 1 GPA', 'Credit Hours', 'Attendance'],
    'Importance': ['32.5%', '27.8%', '18.9%', '11.2%', '4.8%']
})

# 3. Error Analysis Table
errors = pd.DataFrame({
    'Error Range': ['< 0.25 GPA', '0.25-0.50 GPA', '> 0.50 GPA'],
    'Percentage': ['70%', '18%', '12%'],
    'Students': ['70 students', '18 students', '12 students']
})

def create_simple_table(df, title, filename):
    """Create a simple table image."""
    fig, ax = plt.subplots(figsize=(12, len(df)*0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    colColours=['#2c3e50'] * len(df.columns))
    
    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Rows
    for i in range(1, len(df)+1):
        color = '#ecf0f1' if i%2==0 else 'white'
        for j in range(len(df.columns)):
            table[(i, j)].set_facecolor(color)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20, color='#2c3e50')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {filename}")

# Create all tables
create_simple_table(performance, 'Model Performance Summary', 'performance_table.png')
create_simple_table(features, 'Top Predictive Features', 'features_table.png')
create_simple_table(errors, 'Error Distribution Analysis', 'errors_table.png')

print("\n✅ All 3 table images created successfully!")
print("   1. performance_table.png")
print("   2. features_table.png")
print("   3. errors_table.png")
print("\nReady for presentations and documentation!")