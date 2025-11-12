import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

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

class HCTRFEnsemble:
    def __init__(self, hct_model, rf_room_model, rf_time_model, alpha=0.5):
        self.hct_model = hct_model
        self.rf_room_model = rf_room_model
        self.rf_time_model = rf_time_model
        self.alpha = alpha
        
    def _ensure_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, np.ndarray):
            if hasattr(self.rf_room_model, 'feature_names_in_'):
                feature_names = self.rf_room_model.feature_names_in_
            else:
                feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
            return pd.DataFrame(X, columns=feature_names)
        else:
            raise ValueError("Input must be pandas DataFrame or numpy array")
        
    def predict_room_hct(self, node, features):
        if node.is_leaf:
            return node.prediction
        
        if isinstance(features, pd.DataFrame):
            features_array = features.values
        else:
            features_array = features.reshape(1, -1)
        
        cluster_pred = node.classifier.predict(features_array)[0]
        
        if cluster_pred == 1:
            return self.predict_room_hct(node.left, features)
        else:
            return self.predict_room_hct(node.right, features)
    
    def predict_time_hct(self, node, features):
        if node.is_leaf:
            if hasattr(node, 'time_predictor') and node.time_predictor is not None:
                if isinstance(features, pd.DataFrame):
                    features_array = features.values
                else:
                    features_array = features.reshape(1, -1)
                return node.time_predictor.predict(features_array)[0]
            else:
                return 30.0
        
        if isinstance(features, pd.DataFrame):
            features_array = features.values
        else:
            features_array = features.reshape(1, -1)
        
        cluster_pred = node.classifier.predict(features_array)[0]
        
        if cluster_pred == 1:
            return self.predict_time_hct(node.left, features)
        else:
            return self.predict_time_hct(node.right, features)
    
    def predict_room(self, X):
        X_df = self._ensure_dataframe(X)
        predictions = []
        
        for i in range(len(X_df)):
            features = X_df.iloc[i:i+1]
            hct_room_pred = self.predict_room_hct(self.hct_model, features)
            rf_room_pred = self.rf_room_model.predict(features)[0]
            
            if np.random.random() < self.alpha:
                final_pred = hct_room_pred
            else:
                final_pred = rf_room_pred
            
            predictions.append(final_pred)
        
        return np.array(predictions)
    
    def predict_room_proba(self, X):
        X_df = self._ensure_dataframe(X)
        rf_proba = self.rf_room_model.predict_proba(X_df)
        rf_classes = self.rf_room_model.classes_
        hct_proba = np.zeros_like(rf_proba)
        
        for i in range(len(X_df)):
            features = X_df.iloc[i:i+1]
            hct_pred = self.predict_room_hct(self.hct_model, features)
            for j, cls in enumerate(rf_classes):
                if cls == hct_pred:
                    hct_proba[i, j] = 1.0
                    break
        
        ensemble_proba = self.alpha * hct_proba + (1 - self.alpha) * rf_proba
        return ensemble_proba
    
    def predict_time(self, X):
        X_df = self._ensure_dataframe(X)
        predictions = []
        
        for i in range(len(X_df)):
            features = X_df.iloc[i:i+1]
            hct_time_pred = self.predict_time_hct(self.hct_model, features)
            rf_time_pred = self.rf_time_model.predict(features)[0]
            final_pred = self.alpha * hct_time_pred + (1 - self.alpha) * rf_time_pred
            predictions.append(final_pred)
        
        return np.array(predictions)

# -----------------------------
# MULTICLASS CONFUSION MATRIX AND METRICS FUNCTIONS
# -----------------------------
def plot_multiclass_confusion_matrix(y_true, y_pred, model_name, class_names=None):
    """Plot multiclass confusion matrix with metrics"""
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate multiclass metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    error_rate = 1 - accuracy
    
    # Get unique classes
    unique_classes = sorted(set(y_true) | set(y_pred))
    if class_names is None:
        class_names = [f'Class {cls}' for cls in unique_classes]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names,
                yticklabels=class_names)
    ax1.set_title(f'Multiclass Confusion Matrix - {model_name}\n', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('Actual Label', fontsize=12)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    # Plot 2: Metrics Summary
    metrics_data = {
        'Metric': ['Accuracy', 'Error Rate', 
                  'Precision (Micro)', 'Precision (Macro)', 'Precision (Weighted)',
                  'Recall (Micro)', 'Recall (Macro)', 'Recall (Weighted)',
                  'F1-Score (Micro)', 'F1-Score (Macro)', 'F1-Score (Weighted)'],
        'Value': [f'{accuracy:.4f}', f'{error_rate:.4f}',
                 f'{precision_micro:.4f}', f'{precision_macro:.4f}', f'{precision_weighted:.4f}',
                 f'{recall_micro:.4f}', f'{recall_macro:.4f}', f'{recall_weighted:.4f}',
                 f'{f1_micro:.4f}', f'{f1_macro:.4f}', f'{f1_weighted:.4f}']
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    ax2.axis('tight')
    ax2.axis('off')
    table = ax2.table(cellText=metrics_df.values,
                     colLabels=metrics_df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0.1, 0.1, 0.8, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    ax2.set_title('Multiclass Performance Metrics', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed metrics
    print(f"\n📊 DETAILED MULTICLASS METRICS FOR {model_name}:")
    print("=" * 60)
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Error Rate:         {error_rate:.4f}")
    print(f"Precision (Micro):  {precision_micro:.4f}")
    print(f"Precision (Macro):  {precision_macro:.4f}")
    print(f"Precision (Weighted): {precision_weighted:.4f}")
    print(f"Recall (Micro):     {recall_micro:.4f}")
    print(f"Recall (Macro):     {recall_macro:.4f}")
    print(f"Recall (Weighted):  {recall_weighted:.4f}")
    print(f"F1-Score (Micro):   {f1_micro:.4f}")
    print(f"F1-Score (Macro):   {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print("=" * 60)
    
    # Print per-class metrics
    print(f"\n📈 PER-CLASS METRICS FOR {model_name}:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    return {
        'accuracy': accuracy,
        'error_rate': error_rate,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm
    }

def evaluate_all_models_multiclass(ensemble_models, X_test, y_room_test, y_time_test, rf_room_model, hct_model):
    """Evaluate all models with multiclass metrics"""
    
    results = {}
    
    # Get class names from the data
    unique_classes = sorted(y_room_test.unique())
    class_names = [f'Room {cls}' for cls in unique_classes]
    
    # Evaluate ensemble models
    for name, ensemble in ensemble_models.items():
        print(f"\n{'='*70}")
        print(f"EVALUATING {name}")
        print(f"{'='*70}")
        
        # Room predictions
        room_predictions = ensemble.predict_room(X_test)
        
        # Generate confusion matrix and metrics
        metrics = plot_multiclass_confusion_matrix(y_room_test, room_predictions, name, class_names)
        
        # Time prediction evaluation
        time_predictions = ensemble.predict_time(X_test)
        time_mae = mean_absolute_error(y_time_test, time_predictions)
        metrics['time_mae'] = time_mae
        
        print(f"Time Prediction MAE: {time_mae:.4f} seconds")
        results[name] = metrics
    
    # Evaluate individual RF model
    print(f"\n{'='*70}")
    print("EVALUATING RANDOM FOREST ONLY")
    print(f"{'='*70}")
    rf_room_pred = rf_room_model.predict(X_test)
    rf_metrics = plot_multiclass_confusion_matrix(y_room_test, rf_room_pred, "Random Forest Only", class_names)
    rf_time_pred = rf_time_model.predict(X_test)
    rf_metrics['time_mae'] = mean_absolute_error(y_time_test, rf_time_pred)
    print(f"Time Prediction MAE: {rf_metrics['time_mae']:.4f} seconds")
    results['Random Forest Only'] = rf_metrics
    
    # Evaluate individual HCT model
    print(f"\n{'='*70}")
    print("EVALUATING HCT ONLY")
    print(f"{'='*70}")
    
    def predict_hct_individual(hct_model, X_test):
        predictions = []
        for i in range(len(X_test)):
            features = X_test.iloc[i:i+1]
            pred = ensemble_models['Balanced (α=0.5)'].predict_room_hct(hct_model, features)
            predictions.append(pred)
        return np.array(predictions)
    
    hct_room_pred = predict_hct_individual(hct_model, X_test)
    hct_metrics = plot_multiclass_confusion_matrix(y_room_test, hct_room_pred, "HCT Only", class_names)
    
    def predict_hct_time(hct_model, X_test):
        predictions = []
        for i in range(len(X_test)):
            features = X_test.iloc[i:i+1]
            pred = ensemble_models['Balanced (α=0.5)'].predict_time_hct(hct_model, features)
            predictions.append(pred)
        return np.array(predictions)
    
    hct_time_pred = predict_hct_time(hct_model, X_test)
    hct_metrics['time_mae'] = mean_absolute_error(y_time_test, hct_time_pred)
    print(f"Time Prediction MAE: {hct_metrics['time_mae']:.4f} seconds")
    results['HCT Only'] = hct_metrics
    
    return results

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    # Import required for model training if needed
    from sklearn.model_selection import train_test_split
    import os
    
    def train_models_if_needed():
        if (os.path.exists('hct_model_with_time.pkl') and 
            os.path.exists('rf_room_model.pkl') and 
            os.path.exists('rf_time_model.pkl')):
            print("✅ Pre-trained models found!")
            return True
        else:
            print("Models not found. Please train models first.")
            return False
    
    # Check and load models
    if train_models_if_needed():
        print("Loading trained models...")
        hct_model = joblib.load('hct_model_with_time.pkl')
        rf_room_model = joblib.load('rf_room_model.pkl')
        rf_time_model = joblib.load('rf_time_model.pkl')
        X_test_rf, y_room_test_rf, y_time_test_rf = joblib.load('rf_test_data.pkl')
        
        print("✅ Models loaded successfully!")
        
        # Create ensemble models
        ensemble_models = {
            'HCT_Dominant (α=0.7)': HCTRFEnsemble(hct_model, rf_room_model, rf_time_model, alpha=0.7),
            'Balanced (α=0.5)': HCTRFEnsemble(hct_model, rf_room_model, rf_time_model, alpha=0.5),
            'RF_Dominant (α=0.3)': HCTRFEnsemble(hct_model, rf_room_model, rf_time_model, alpha=0.3)
        }
        
        # Evaluate all models with comprehensive multiclass metrics
        print("\n🚀 STARTING COMPREHENSIVE MULTICLASS MODEL EVALUATION")
        print("=" * 70)
        
        results = evaluate_all_models_multiclass(ensemble_models, X_test_rf, y_room_test_rf, y_time_test_rf, 
                                               rf_room_model, hct_model)
        
        # Find best ensemble based on weighted F1-score and time MAE
        best_ensemble_name = max(ensemble_models.keys(), 
                                key=lambda x: results[x]['f1_weighted'] - results[x]['time_mae']/100)
        best_ensemble = ensemble_models[best_ensemble_name]
        
        print(f"\n🎯 BEST ENSEMBLE CONFIGURATION: {best_ensemble_name}")
        print(f"   Accuracy: {results[best_ensemble_name]['accuracy']:.4f}")
        print(f"   F1-Score (Weighted): {results[best_ensemble_name]['f1_weighted']:.4f}")
        print(f"   Time MAE: {results[best_ensemble_name]['time_mae']:.4f} seconds")
        
        # Save best ensemble
        joblib.dump(best_ensemble, 'hct_rf_ensemble_model.pkl')
        print("\n✅ Best ensemble model saved as 'hct_rf_ensemble_model.pkl'")
        
        # Create comprehensive comparison table
        print(f"\n{'='*90}")
        print("COMPREHENSIVE MULTICLASS MODEL COMPARISON")
        print(f"{'='*90}")
        
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'F1-Macro': f"{metrics['f1_macro']:.4f}",
                'F1-Weighted': f"{metrics['f1_weighted']:.4f}",
                'Precision-Macro': f"{metrics['precision_macro']:.4f}",
                'Recall-Macro': f"{metrics['recall_macro']:.4f}",
                'Error Rate': f"{metrics['error_rate']:.4f}",
                'Time MAE': f"{metrics['time_mae']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Plot overall comparison
        plt.figure(figsize=(14, 8))
        models = list(results.keys())
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Accuracy and F1-Scores
        accuracy_scores = [results[model]['accuracy'] for model in models]
        f1_weighted_scores = [results[model]['f1_weighted'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, accuracy_scores, width, label='Accuracy', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, f1_weighted_scores, width, label='F1-Weighted', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Scores')
        ax1.set_title('Model Comparison: Accuracy vs F1-Weighted Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Precision and Recall (Macro)
        precision_scores = [results[model]['precision_macro'] for model in models]
        recall_scores = [results[model]['recall_macro'] for model in models]
        
        bars3 = ax2.bar(x - width/2, precision_scores, width, label='Precision (Macro)', alpha=0.8, color='lightgreen')
        bars4 = ax2.bar(x + width/2, recall_scores, width, label='Recall (Macro)', alpha=0.8, color='gold')
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Scores')
        ax2.set_title('Model Comparison: Precision vs Recall (Macro)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Time MAE
        time_mae_scores = [results[model]['time_mae'] for model in models]
        
        bars5 = ax3.bar(x, time_mae_scores, width, label='Time MAE', alpha=0.8, color='orange')
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('MAE (seconds)')
        ax3.set_title('Model Comparison: Time Prediction MAE')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        for bar in bars5:
            height = bar.get_height()
            ax3.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Error Rate
        error_rates = [results[model]['error_rate'] for model in models]
        
        bars6 = ax4.bar(x, error_rates, width, label='Error Rate', alpha=0.8, color='red')
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Error Rate')
        ax4.set_title('Model Comparison: Error Rate')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        for bar in bars6:
            height = bar.get_height()
            ax4.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
    else:
        print("❌ Please run the training scripts first to generate the model files.")