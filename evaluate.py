import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import joblib
import json

# dans evaluate.py
from pathlib import Path
Path("reports").mkdir(parents=True, exist_ok=True)
# ... puis plt.savefig('reports/confusion_matrix.png', ...)


# Charger les donnees (meme split que train.py)
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Charger le modele
model = joblib.load('models/iris_model.pkl')

# Predictions
y_pred = model.predict(X_test)

# 1. Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.title('Confusion Matrix - Iris Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('reports/confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.close()

# 2. Feature importance
feature_importance = model.feature_importances_
plt.figure(figsize=(10, 6))
indices = np.argsort(feature_importance)[::-1]
plt.bar(range(len(feature_importance)), 
        feature_importance[indices])
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(range(len(feature_importance)), 
           [iris.feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig('reports/feature_importance.png', dpi=120, bbox_inches='tight')
plt.close()

print("Visualizations generated successfully!")
print("- reports/confusion_matrix.png")
print("- reports/feature_importance.png")

