import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv('synthetic_indoor_localization_data.csv')

# If Next_Time is not in dataset, simulate realistic movement duration (in seconds)
if 'Next_Time' not in data.columns:
    np.random.seed(42)
    data['Next_Time'] = np.random.uniform(10, 90, len(data))

# Separate features and labels
X = data.drop(columns=['Timestamp', 'Next_Room', 'Next_Time'])
y_room = data['Next_Room']
y_time = data['Next_Time']
X_np = X.to_numpy()

# Train/test split
train_idx, test_idx = train_test_split(range(len(y_room)), test_size=0.3, random_state=42)

# -----------------------------
# DEFINE HCT NODE CLASS
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
# RECURSIVE HCT BUILDER
# -----------------------------
def build_hct_node(indices, max_depth=4, min_size=50, depth=0):
    node = HCTNode(indices, depth)

    # Stop condition
    if depth >= max_depth or len(indices) <= min_size:
        node.is_leaf = True
        node.prediction = y_room.iloc[indices].mode()[0]
        return node

    # Hierarchical clustering
    sub_X = X_np[indices]
    sub_linkage = linkage(sub_X, method='ward')
    clusters = fcluster(sub_linkage, 2, criterion='maxclust')

    # Split indices
    left_indices = [indices[i] for i in range(len(indices)) if clusters[i] == 1]
    right_indices = [indices[i] for i in range(len(indices)) if clusters[i] == 2]

    # If clustering fails, make leaf
    if len(left_indices) == 0 or len(right_indices) == 0:
        node.is_leaf = True
        node.prediction = y_room.iloc[indices].mode()[0]
        return node

    # Train Random Forest classifier (room cluster)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(sub_X, clusters)
    node.classifier = clf

    # Train Random Forest regressor (travel time)
    sub_y_time = y_time.iloc[indices].values
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(sub_X, sub_y_time)
    node.time_predictor = reg

    # Recursively build children
    node.left = build_hct_node(left_indices, max_depth, min_size, depth + 1)
    node.right = build_hct_node(right_indices, max_depth, min_size, depth + 1)

    return node


# -----------------------------
# TRAIN MODEL
# -----------------------------
print("Training HCT model... This may take a few minutes.")
root_node = build_hct_node(train_idx)

# Save trained model and test indices
joblib.dump(root_node, 'hct_model_with_time.pkl')
joblib.dump(test_idx, 'test_indices_with_time.pkl')

print("✅ Training complete. Model and test indices saved as:")
print("   - hct_model_with_time.pkl")
print("   - test_indices_with_time.pkl")
