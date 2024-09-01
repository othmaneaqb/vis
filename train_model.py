# train_new_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# New training data
new_training_data = {
    'num_columns': [2, 3, 5, 4, 3],
    'num_rows': [200, 300, 150, 400, 250],
    'num_unique': [15, 25, 40, 35, 20],
    'variance': [2.0, 1.2, 0.8, 1.5, 1.1],
    'skewness': [0.3, -0.1, 0.2, 0.4, -0.3],
    'kurtosis': [1.1, -0.6, 0.8, -0.4, 0.5],
    'missing_values': [1, 3, 0, 2, 1],
    'recommended_visualization': ['scatter', 'line', 'heatmap', 'bar', 'box']
}

df = pd.DataFrame(new_training_data)

# Encode target labels
le = LabelEncoder()
df['recommended_visualization'] = le.fit_transform(df['recommended_visualization'])

# Split data
X = df.drop(columns=['recommended_visualization'])
y = df['recommended_visualization']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")

# Save model and label encoder
joblib.dump(model, 'visualization_recommender.pkl')
np.save('new_label_classes.npy', le.classes_)
