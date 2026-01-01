# create_test_split.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load your cleaned data
df = pd.read_csv("data/cleaned_student_performance.csv")
print(f"Original data shape: {df.shape}")

# Check if we have the target column
if 'Final_Year_GPA' in df.columns:
    # Split into train and test (80% train, 20% test)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save train and test sets
    train_df.to_csv("data/train_students.csv", index=False)
    test_df.to_csv("data/test_students.csv", index=False)
    
    print(f"✅ Created train data: {train_df.shape}")
    print(f"✅ Created test data: {test_df.shape}")
    print(f"Test data saved to: data/test_students.csv")
else:
    print("❌ 'Final_Year_GPA' column not found in data")
    print("Available columns:", df.columns.tolist())