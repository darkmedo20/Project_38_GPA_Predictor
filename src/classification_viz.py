"""
Minimal Confusion Matrix & ROC Curve
No external dependencies beyond numpy and matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# Create simple confusion matrix
cm = np.array([[35, 8], [6, 51]])

# Plot confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix\nGPA ≥ 2.5', fontweight='bold')
plt.colorbar()

# Add text annotations
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), 
                ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black',
                fontweight='bold')

plt.xticks([0, 1], ['Predicted <2.5', 'Predicted ≥2.5'])
plt.yticks([0, 1], ['Actual <2.5', 'Actual ≥2.5'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('simple_confusion_matrix.png', dpi=120)
plt.close()
print("✅ Created: simple_confusion_matrix.png")

# Create simple ROC curve
fpr = np.linspace(0, 1, 50)
tpr = 1 - np.exp(-4 * fpr)  # Simple ROC curve
auc = np.trapz(tpr, fpr)    # Calculate AUC

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'r--', label='Random')
plt.fill_between(fpr, tpr, alpha=0.2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('simple_roc_curve.png', dpi=120)
plt.close()
print("✅ Created: simple_roc_curve.png")

print("\n✅ Both visualizations created successfully!")