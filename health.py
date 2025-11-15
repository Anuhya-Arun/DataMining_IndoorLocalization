import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, mean_absolute_error

# =====================================================
#  1. HCT Node Class + Prediction Function
# =====================================================
class HCTNode:
    def _init_(self, indices=None, depth=0):
        self.indices = indices
        self.left = None
        self.right = None
        self.classifier = None
        self.time_predictor = None
        self.depth = depth
        self.is_leaf = False
        self.prediction = None


def hct_predict(node, x):
    """Recursive HCT prediction for next room + time."""
    if node.is_leaf:
        return node.prediction, None

    pred = node.classifier.predict(x.reshape(1, -1))[0]

    if pred == 1 and node.left:
        room, _ = hct_predict(node.left, x)
    elif pred == 2 and node.right:
        room, _ = hct_predict(node.right, x)
    else:
        return node.prediction, None

    time_pred = None
    if node.time_predictor:
        time_pred = node.time_predictor.predict(x.reshape(1, -1))[0]

    return room, time_pred


# =====================================================
#  2. Load Data + Model
# =====================================================

csv=r"E:\DATA MINING dataset\new data\synthetic_indoor_localization_data.csv"
data = pd.read_csv(csv)

# Create Next_Time if missing
if 'Next_Time' not in data.columns:
    np.random.seed(42)
    data['Next_Time'] = np.random.uniform(10, 90, len(data))

X = data.drop(columns=['Timestamp', 'Next_Room', 'Next_Time'])
y_true_room = data['Next_Room']
y_true_time = data['Next_Time']
X_np = X.to_numpy()

# Load pretrained HCT model + test indices
hct_model = joblib.load('hct_model_with_time.pkl')
test_idx = joblib.load('test_indices_with_time.pkl')

# =====================================================
#  3. Evaluate Model
# =====================================================
pred_rooms, pred_times = [], []

for x in X_np[test_idx]:
    room_pred, time_pred = hct_predict(hct_model, x)
    pred_rooms.append(room_pred)
    pred_times.append(time_pred if time_pred is not None else np.nan)

acc = accuracy_score(y_true_room.iloc[test_idx], pred_rooms)
mae = mean_absolute_error(y_true_time.iloc[test_idx], pred_times)

print(f"\n HCT Evaluation Results:")
print(f"   - Next Room Accuracy: {acc:.4f}")
print(f"   - Mean Absolute Error (Travel Time): {mae:.2f} seconds")

# =====================================================
#  4. Simulate Actual Times (replace with real later)
# =====================================================
# Add random deviation to predicted times
np.random.seed(42)
actual_times = np.array(pred_times) + np.random.uniform(-10, 30, len(pred_times))
actual_times = np.clip(actual_times, 5, None)  # no negative or too small times

# =====================================================
#  5. Health Status Detection
# =====================================================
health_status = []
for pred_t, act_t in zip(pred_times, actual_times):
    if np.isnan(pred_t):
        health_status.append(" No Prediction")
    elif act_t <= pred_t + 5:
        health_status.append(" Normal")
    elif act_t <= pred_t * 1.5:
        health_status.append(" Slight Delay")
    else:
        health_status.append(" Needs Help")

health_df = pd.DataFrame({
    "Predicted_Room": pred_rooms,
    "Actual_Room": y_true_room.iloc[test_idx].values,
    "Predicted_Time": pred_times,
    "Actual_Time": actual_times,
    "Health_Status": health_status
})

# Save results
health_df.to_csv("person_health_status.csv", index=False)
print("\n Health status report saved as 'person_health_status.csv'")

# =====================================================
# ✅ 6. Visualization
# =====================================================
plt.figure(figsize=(10, 6))
colors = {' Normal': 'green', ' Slight Delay': 'gold', ' Needs Help': 'red', ' No Prediction': 'gray'}

plt.scatter(
    range(len(health_status)),
    actual_times,
    c=[colors.get(h, 'gray') for h in health_status],
    s=80
)

plt.xlabel("Movement Instance")
plt.ylabel("Actual Travel Time (s)")
plt.title(f"Movement Health Detection (Acc: {acc:.2f}, MAE: {mae:.2f}s)")
plt.grid(True)
plt.show()

# =====================================================
#  7. Summary Statistics
# =====================================================
print("\nSummary of movement health:")
print(health_df['Health_Status'].value_counts())