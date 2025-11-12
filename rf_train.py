import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = pd.read_csv('synthetic_indoor_localization_data.csv')

# Feature matrix
X = data.drop(columns=['Timestamp', 'Next_Room', 'Next_Time'])

# Classification target: next room
y_room = data['Next_Room']

# Regression target: time taken to reach next room
y_time = data['Next_Time']

# Split into train/test sets (consistent split)
X_train, X_test, y_room_train, y_room_test, y_time_train, y_time_test = train_test_split(
    X, y_room, y_time, test_size=0.3, random_state=42
)

# Train Random Forest for room prediction
print("Training Random Forest (Room Prediction)...")
clf_room = RandomForestClassifier(n_estimators=100, random_state=42)
clf_room.fit(X_train, y_room_train)

# Train Random Forest for time prediction
print("Training Random Forest (Time Prediction)...")
clf_time = RandomForestRegressor(n_estimators=100, random_state=42)
clf_time.fit(X_train, y_time_train)

# Save trained models and test sets
joblib.dump(clf_room, 'rf_room_model.pkl')
joblib.dump(clf_time, 'rf_time_model.pkl')
joblib.dump((X_test, y_room_test, y_time_test), 'rf_test_data.pkl')

print("Training complete. Both models and test data saved successfully.")
