import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, mean_absolute_error

# -----------------------------
# DEFINE NODE CLASS
# -----------------------------
class HCTNode:
    def __init__(self, indices, depth=0):
        self.indices = indices
        self.left = None
        self.right = None
        self.classifier = None
        self.time_predictor = None
        self.depth = depth
        self.is_leaf = False
        self.prediction = None


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def hct_predict(node, x):
    if node.is_leaf:
        return node.prediction, None  # no time predictor at leaf

    pred = node.classifier.predict(x.reshape(1, -1))[0]

    # Recurse based on cluster prediction
    if pred == 1:
        room, _ = hct_predict(node.left, x)
    else:
        room, _ = hct_predict(node.right, x)

    # Predict time at current node (if available)
    time_pred = None
    if node.time_predictor:
        time_pred = node.time_predictor.predict(x.reshape(1, -1))[0]

    return room, time_pred


# -----------------------------
# LOAD DATA AND MODEL
# -----------------------------
data = pd.read_csv('synthetic_indoor_localization_data.csv')

# Ensure Next_Time exists (for evaluation)
if 'Next_Time' not in data.columns:
    np.random.seed(42)
    data['Next_Time'] = np.random.uniform(10, 90, len(data))

X = data.drop(columns=['Timestamp', 'Next_Room', 'Next_Time'])
y_true_room = data['Next_Room']
y_true_time = data['Next_Time']
X_np = X.to_numpy()

# Load model and test indices
hct_model = joblib.load('hct_model_with_time.pkl')
test_idx = joblib.load('test_indices_with_time.pkl')

# -----------------------------
# EVALUATE MODEL
# -----------------------------
pred_rooms = []
pred_times = []

for x in X_np[test_idx]:
    room_pred, time_pred = hct_predict(hct_model, x)
    pred_rooms.append(room_pred)
    pred_times.append(time_pred if time_pred is not None else np.nan)

# Compute metrics
acc = accuracy_score(y_true_room.iloc[test_idx], pred_rooms)
mae = mean_absolute_error(y_true_time.iloc[test_idx], pred_times)

print(f"✅ HCT Evaluation Results:")
print(f"   - Next Room Accuracy: {acc:.4f}")
print(f"   - Mean Absolute Error (Travel Time): {mae:.2f} seconds")

# -----------------------------
# VISUALIZE RESULTS
# -----------------------------
plt.ion()
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(pred_rooms[:50], 'r--o', label='Predicted Room')
ax1.plot(y_true_room.iloc[test_idx[:50]].values, 'g-', label='Actual Room')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Room Number')
ax1.set_title(f"Indoor Localization (Acc: {acc:.2f}, Time MAE: {mae:.2f}s)")
ax1.legend(loc='upper right')

plt.show(block=True)
plt.ioff()
print("Visualization complete.")
