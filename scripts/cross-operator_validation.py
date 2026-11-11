import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                            roc_curve, precision_recall_curve,
                            accuracy_score, precision_score, recall_score, f1_score,
                            matthews_corrcoef, cohen_kappa_score)
from sklearn.calibration import calibration_curve
import pandas as pd
# Set non-interactive backend to prevent display issues
import matplotlib
matplotlib.use('Agg')

def load_cross_operator_dataset(dataset_path):
    """
    Load cross-operator validation dataset from specified directory structure.
    Args:
        dataset_path (str): Path to cross_operator_validation_dataset/test/
    Returns:
        tuple: (X_cross_operator, y_cross_operator) - Images and labels as numpy arrays
    """
    print("Loading cross-operator validation dataset...")
    normal_path = os.path.join(dataset_path, "NORMAL")
    pneumonia_path = os.path.join(dataset_path, "PNEUMONIA")
    images = []
    labels = []
    # Validate paths exist
    if not os.path.exists(normal_path):
        raise FileNotFoundError(f"Normal images path not found: {normal_path}")
    if not os.path.exists(pneumonia_path):
        raise FileNotFoundError(f"Pneumonia images path not found: {pneumonia_path}")
    # Load normal images (label = 0)
    print(f"Loading normal images from: {normal_path}")
    for filename in os.listdir(normal_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(normal_path, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(0)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    # Load pneumonia images (label = 1)
    print(f"Loading pneumonia images from: {pneumonia_path}")
    for filename in os.listdir(pneumonia_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(pneumonia_path, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(1)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    if len(images) == 0:
        raise ValueError("No images were loaded. Check dataset path and image formats.")
    print(f"Loaded {len(images)} cross-operator validation samples")
    print(f"Normal: {labels.count(0)}, Pneumonia: {labels.count(1)}")
    return np.array(images), np.array(labels)

def calculate_clinical_metrics(y_true, y_pred):
    """Calculate comprehensive clinical metrics for binary classification."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred) # Sensitivity
    f1 = f1_score(y_true, y_pred)
    # Clinical metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0 # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0 # Negative Predictive Value
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0 # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0 # False Negative Rate
    # Advanced metrics
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': recall,
        'specificity': specificity,
        'f1': f1,
        'ppv': ppv,
        'npv': npv,
        'fpr': fpr,
        'fnr': fnr,
        'mcc': mcc,
        'kappa': kappa,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }

def generate_comprehensive_visualizations(y_true, y_pred, y_prob, metrics):
    """Generate comprehensive visualization suite for cross-operator validation."""
    plt.style.use('default')
    sns.set_palette("husl")
    # 1. Enhanced Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annot_labels = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1%})'
                              for j in range(cm.shape[1])]
                             for i in range(cm.shape[0])])
    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues',
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'],
                cbar_kws={'label': 'Count'})
    plt.title('Cross-Operator Validation - Enhanced Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('1_enhanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 2. ROC Curve
    fpr_vals, tpr_vals, _ = roc_curve(y_true, y_prob.flatten())
    roc_auc = roc_auc_score(y_true, y_prob.flatten())
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_vals, tpr_vals, color='darkorange', lw=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Cross-Operator Validation - ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('2_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 3. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob.flatten())
    pr_auc = abs(np.trapz(precision_vals, recall_vals))
    plt.figure(figsize=(10, 8))
    plt.plot(recall_vals, precision_vals, color='blue', lw=3, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.axhline(y=np.mean(y_true), color='red', linestyle='--',
                label=f'Baseline (Prevalence = {np.mean(y_true):.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision (PPV)', fontsize=12)
    plt.title('Cross-Operator Validation - Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('3_precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 4. Performance Comparison Chart
    internal_results = {'accuracy': 0.948, 'sensitivity': 0.896, 'specificity': 1.000}
    external_vals = [metrics['accuracy'], metrics['sensitivity'], metrics['specificity']]
    internal_vals = [internal_results['accuracy'], internal_results['sensitivity'], internal_results['specificity']]
    metric_names = ['Accuracy', 'Sensitivity', 'Specificity']
    x_pos = np.arange(len(metric_names))
    width = 0.35
    plt.figure(figsize=(12, 8))
    bars1 = plt.bar(x_pos - width/2, internal_vals, width, label='Internal Validation',
                    color='lightblue', alpha=0.8, edgecolor='darkblue')
    bars2 = plt.bar(x_pos + width/2, external_vals, width, label='Cross-Operator Validation',
                    color='orange', alpha=0.8, edgecolor='darkorange')
    # Add value labels
    for i, (internal, external) in enumerate(zip(internal_vals, external_vals)):
        plt.text(i - width/2, internal + 0.01, f'{internal:.3f}', ha='center', va='bottom', fontweight='bold')
        plt.text(i + width/2, external + 0.01, f'{external:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Performance Score', fontsize=12)
    plt.title('Internal vs Cross-Operator Validation - Performance Comparison', fontsize=16, fontweight='bold')
    plt.xticks(x_pos, metric_names)
    plt.legend(fontsize=12)
    plt.ylim([0, 1.1])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('4_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 5. Class Distribution
    class_counts = [np.sum(y_true == 0), np.sum(y_true == 1)]
    class_labels = ['Normal', 'Pneumonia']
    colors = ['lightblue', 'lightcoral']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    bars = ax1.bar(class_labels, class_counts, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Cross-Operator Dataset - Class Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars, class_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(count), ha='center', va='bottom', fontweight='bold')
    ax2.pie(class_counts, labels=class_labels, colors=colors, autopct='%1.1f%%',
            startangle=90, explode=(0.05, 0.05))
    ax2.set_title('Cross-Operator Dataset - Class Proportion', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('5_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 6. Prediction Confidence Distribution
    plt.figure(figsize=(12, 8))
    normal_preds = y_prob[y_true == 0].flatten()
    pneumonia_preds = y_prob[y_true == 1].flatten()
    plt.hist(normal_preds, bins=30, alpha=0.7, label='Normal Cases', color='lightblue', density=True)
    plt.hist(pneumonia_preds, bins=30, alpha=0.7, label='Pneumonia Cases', color='lightcoral', density=True)
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    plt.xlabel('Prediction Confidence (Probability)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Cross-Operator Validation - Prediction Confidence Distribution', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('6_prediction_confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 7. Calibration Plot
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob.flatten(), n_bins=10)
        plt.figure(figsize=(10, 8))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", linewidth=2, label='Model Calibration')
        plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Cross-Operator Validation - Calibration Plot', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('7_calibration_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate calibration plot: {e}")
    # 8. Comprehensive Metrics Dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # Sensitivity vs Specificity
    axes[0,0].bar(['Sensitivity', 'Specificity'], [metrics['sensitivity'], metrics['specificity']],
                  color=['green', 'blue'], alpha=0.7)
    axes[0,0].set_ylim([0, 1])
    axes[0,0].set_title('Sensitivity vs Specificity', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    # PPV vs NPV
    axes[0,1].bar(['PPV', 'NPV'], [metrics['ppv'], metrics['npv']], color=['orange', 'purple'], alpha=0.7)
    axes[0,1].set_ylim([0, 1])
    axes[0,1].set_title('Positive vs Negative Predictive Value', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    # Error Rates
    max_error = max(metrics['fpr'], metrics['fnr']) if max(metrics['fpr'], metrics['fnr']) > 0 else 0.1
    axes[0,2].bar(['FPR', 'FNR'], [metrics['fpr'], metrics['fnr']], color=['red', 'darkred'], alpha=0.7)
    axes[0,2].set_ylim([0, max_error * 1.2])
    axes[0,2].set_title('False Positive vs False Negative Rate', fontweight='bold')
    axes[0,2].grid(True, alpha=0.3)
    # Advanced Metrics
    axes[1,0].bar(['F1-Score', 'MCC', 'Kappa'], [metrics['f1'], metrics['mcc'], metrics['kappa']],
                  color=['cyan', 'magenta', 'yellow'], alpha=0.7)
    axes[1,0].set_ylim([-1, 1])
    axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1,0].set_title('Advanced Performance Metrics', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    # ROC-AUC vs PR-AUC
    axes[1,1].bar(['ROC-AUC', 'PR-AUC'], [roc_auc, pr_auc],
                  color=['darkblue', 'darkgreen'], alpha=0.7)
    axes[1,1].set_ylim([0, 1])
    axes[1,1].set_title('Area Under Curve Metrics', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    # Confusion Matrix Breakdown
    cm_values = [metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn']]
    cm_labels = ['TP', 'TN', 'FP', 'FN']
    colors_cm = ['green', 'blue', 'orange', 'red']
    axes[1,2].bar(cm_labels, cm_values, color=colors_cm, alpha=0.7)
    axes[1,2].set_title('Confusion Matrix Breakdown', fontweight='bold')
    axes[1,2].grid(True, alpha=0.3)
    plt.suptitle('Cross-Operator Validation - Comprehensive Metrics Dashboard',
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('8_comprehensive_metrics_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_cross_operator_validation(model_path, cross_operator_data_path):
    """
    Comprehensive cross-operator validation evaluation with metrics and visualizations.
    Args:
        model_path (str): Path to trained model file
        cross_operator_data_path (str): Path to cross-operator validation dataset
    Returns:
        dict: Complete validation results and metrics
    """
    # Validate model path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # Load trained model
    print("Loading trained model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")
    # Load cross-operator dataset
    X_cross_operator, y_cross_operator = load_cross_operator_dataset(cross_operator_data_path)
    # Generate predictions
    print("Running predictions on cross-operator dataset...")
    predictions_prob = model.predict(X_cross_operator)
    predictions_binary = (predictions_prob > 0.5).astype(int).flatten()
    # Calculate comprehensive metrics
    metrics = calculate_clinical_metrics(y_cross_operator, predictions_binary)
    roc_auc = roc_auc_score(y_cross_operator, predictions_prob.flatten())
    # Print results
    print("\n" + "="*60)
    print("COMPREHENSIVE CROSS-OPERATOR VALIDATION RESULTS")
    print("="*60)
    print(f"Dataset Size: {len(y_cross_operator)} samples")
    print(f"Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"Sensitivity (Recall/TPR): {metrics['sensitivity']:.3f} ({metrics['sensitivity']*100:.1f}%)")
    print(f"Specificity (TNR): {metrics['specificity']:.3f} ({metrics['specificity']*100:.1f}%)")
    print(f"Precision (PPV): {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"Negative Predictive Value: {metrics['npv']:.3f} ({metrics['npv']*100:.1f}%)")
    print(f"F1-Score: {metrics['f1']:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"Matthews Correlation Coefficient: {metrics['mcc']:.3f}")
    print(f"Cohen's Kappa: {metrics['kappa']:.3f}")
    print(f"False Positive Rate: {metrics['fpr']:.3f} ({metrics['fpr']*100:.1f}%)")
    print(f"False Negative Rate: {metrics['fnr']:.3f} ({metrics['fnr']*100:.1f}%)")
    print(f"\nConfusion Matrix Breakdown:")
    print(f"True Positives (Pneumonia correctly identified): {metrics['tp']}")
    print(f"True Negatives (Normal correctly identified): {metrics['tn']}")
    print(f"False Positives (Normal incorrectly flagged): {metrics['fp']}")
    print(f"False Negatives (Pneumonia missed): {metrics['fn']}")
    # Internal vs Cross-Operator comparison
    internal_results = {'accuracy': 0.948, 'sensitivity': 0.896, 'specificity': 1.000, 'dataset_size': 269}
    print("\n" + "="*60)
    print("INTERNAL vs CROSS-OPERATOR COMPARISON")
    print("="*60)
    print(f"{'Metric':<15} {'Internal':<12} {'Cross-Operator':<12} {'Difference':<12}")
    print("-" * 55)
    print(f"{'Accuracy':<15} {internal_results['accuracy']:<12.3f} {metrics['accuracy']:<12.3f} {internal_results['accuracy'] - metrics['accuracy']:<+12.3f}")
    print(f"{'Sensitivity':<15} {internal_results['sensitivity']:<12.3f} {metrics['sensitivity']:<12.3f} {internal_results['sensitivity'] - metrics['sensitivity']:<+12.3f}")
    print(f"{'Specificity':<15} {internal_results['specificity']:<12.3f} {metrics['specificity']:<12.3f} {internal_results['specificity'] - metrics['specificity']:<+12.3f}")
    print(f"{'Sample Size':<15} {internal_results['dataset_size']:<12} {len(y_cross_operator):<12} {len(y_cross_operator) - internal_results['dataset_size']:<+12}")
    # Performance analysis
    acc_drop = internal_results['accuracy'] - metrics['accuracy']
    print(f"\nPerformance Analysis:")
    if acc_drop <= 0.05:
        print(f"Excellent generalization (accuracy drop: {acc_drop:.3f})")
    elif acc_drop <= 0.10:
        print(f"Good generalization (accuracy drop: {acc_drop:.3f})")
    elif acc_drop <= 0.20:
        print(f"Moderate generalization (accuracy drop: {acc_drop:.3f})")
    else:
        print(f"Significant performance drop (accuracy drop: {acc_drop:.3f}) - may indicate overfitting")
    # Generate comprehensive visualizations
    print("\nGenerating comprehensive visualization suite...")
    generate_comprehensive_visualizations(y_cross_operator, predictions_binary, predictions_prob, metrics)
    # Save comprehensive results
    all_metrics = {
        'cross_operator_accuracy': metrics['accuracy'],
        'cross_operator_sensitivity': metrics['sensitivity'],
        'cross_operator_specificity': metrics['specificity'],
        'cross_operator_precision': metrics['precision'],
        'cross_operator_f1': metrics['f1'],
        'cross_operator_roc_auc': roc_auc,
        'cross_operator_mcc': metrics['mcc'],
        'cross_operator_kappa': metrics['kappa'],
        'cross_operator_ppv': metrics['ppv'],
        'cross_operator_npv': metrics['npv'],
        'cross_operator_fpr': metrics['fpr'],
        'cross_operator_fnr': metrics['fnr'],
        'cross_operator_samples': len(y_cross_operator),
        'accuracy_drop': acc_drop,
        'true_positives': metrics['tp'],
        'true_negatives': metrics['tn'],
        'false_positives': metrics['fp'],
        'false_negatives': metrics['fn']
    }
    results_df = pd.DataFrame([all_metrics])
    results_df.to_csv('comprehensive_cross_operator_validation_results.csv', index=False)
    # Generate classification report
    class_report = classification_report(y_cross_operator, predictions_binary,
                                        target_names=['Normal', 'Pneumonia'],
                                        output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_df.to_csv('classification_report.csv')
    print("\nCross-operator validation completed successfully")
    print("Files generated:")
    print(" comprehensive_cross_operator_validation_results.csv")
    print(" classification_report.csv")
    print(" 8 comprehensive visualization files")
    return all_metrics

if __name__ == "__main__":
    # Configuration
    model_path = "../models/best_chest_xray_model.h5"
    cross_operator_data_path = "../cross_operator_validation_dataset/test"
    # Execute cross-operator validation
    results = evaluate_cross_operator_validation(model_path, cross_operator_data_path)
