"""
Fixed Model Evaluation Script - JSON Serialization Issue Resolved
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score
)
from pathlib import Path
import json

class ModelEvaluator:
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.test_generator = None
        
    def load_model_and_data(self):
        """Load trained model and test data"""
        
        print("Loading model and test data...")
        
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from: {self.model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.test_generator = test_datagen.flow_from_directory(
            self.data_path / 'test',
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            shuffle=False
        )
        
        print(f"Test data loaded: {self.test_generator.samples} samples")
        print(f"Class mapping: {self.test_generator.class_indices}")
        
        return True
    
    def generate_predictions(self):
        """Generate predictions and return all necessary arrays"""
        
        print("\nGenerating predictions...")
        
        self.test_generator.reset()
        predictions_prob = self.model.predict(self.test_generator, verbose=0)
        predictions_prob = predictions_prob.flatten()
        predictions_binary = (predictions_prob > 0.5).astype(int)
        true_labels = self.test_generator.classes
        
        print(f"Generated {len(predictions_prob)} predictions")
        
        return predictions_prob, predictions_binary, true_labels
    
    def calculate_all_metrics(self, predictions_prob, predictions_binary, true_labels):
        """Calculate comprehensive evaluation metrics"""
        
        print("\nCalculating comprehensive metrics...")
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions_binary)
        precision = precision_score(true_labels, predictions_binary)
        recall = recall_score(true_labels, predictions_binary)
        f1 = f1_score(true_labels, predictions_binary)
        
        # ROC and PR AUC
        roc_auc = roc_auc_score(true_labels, predictions_prob)
        pr_auc = average_precision_score(true_labels, predictions_prob)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions_binary)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # FIXED: Convert numpy types to Python types immediately
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'specificity': float(specificity),
            'sensitivity': float(sensitivity),
            'false_positive_rate': float(false_positive_rate),
            'false_negative_rate': float(false_negative_rate),
            'tn': int(tn),
            'fp': int(fp), 
            'fn': int(fn),
            'tp': int(tp),
            'confusion_matrix': cm.tolist()  # Convert numpy array to list
        }
    
    def print_comprehensive_report(self, metrics):
        """Print detailed evaluation report"""
        
        print("\n" + "="*80)
        print("CHEST X-RAY PNEUMONIA DETECTION - COMPREHENSIVE EVALUATION REPORT")
        print("="*80)
        
        print(f"\nOVERALL PERFORMANCE METRICS:")
        print("-" * 50)
        print(f"Overall Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']:.1%})")
        print(f"ROC AUC Score:          {metrics['roc_auc']:.4f}")
        print(f"PR AUC Score:           {metrics['pr_auc']:.4f}")
        print(f"Overall Precision:      {metrics['precision']:.4f}")
        print(f"Overall Recall:         {metrics['recall']:.4f}")
        print(f"Overall F1-Score:       {metrics['f1_score']:.4f}")
        
        print(f"\nCONFUSION MATRIX BREAKDOWN:")
        print("-" * 50)
        print(f"True Negatives (TN):    {metrics['tn']} (Correct Normal predictions)")
        print(f"False Positives (FP):   {metrics['fp']} (Normal predicted as Pneumonia)")
        print(f"False Negatives (FN):   {metrics['fn']} (Pneumonia predicted as Normal)")
        print(f"True Positives (TP):    {metrics['tp']} (Correct Pneumonia predictions)")
        
        print(f"\nMEDICAL/CLINICAL METRICS:")
        print("-" * 50)
        print(f"Sensitivity (TPR):      {metrics['sensitivity']:.4f} ({metrics['sensitivity']:.1%})")
        print(f"Specificity (TNR):      {metrics['specificity']:.4f} ({metrics['specificity']:.1%})")
        print(f"False Positive Rate:    {metrics['false_positive_rate']:.4f} ({metrics['false_positive_rate']:.1%})")
        print(f"False Negative Rate:    {metrics['false_negative_rate']:.4f} ({metrics['false_negative_rate']:.1%})")
        
        # Performance assessment
        if metrics['accuracy'] >= 0.95:
            performance = "OUTSTANDING"
        elif metrics['accuracy'] >= 0.90:
            performance = "EXCELLENT"
        elif metrics['accuracy'] >= 0.85:
            performance = "GOOD"
        else:
            performance = "NEEDS IMPROVEMENT"
            
        print(f"\nPERFORMANCE ASSESSMENT:")
        print("-" * 50)
        print(f"Overall Assessment:     {performance}")
        
        clinical_ready = (
            metrics['accuracy'] >= 0.90 and 
            metrics['sensitivity'] >= 0.85 and 
            metrics['specificity'] >= 0.85
        )
        
        clinical_status = "READY for clinical validation" if clinical_ready else "NEEDS improvement"
        print(f"Clinical Readiness:     {clinical_status}")
        
        total_samples = metrics['tn'] + metrics['fp'] + metrics['fn'] + metrics['tp']
        print(f"\nEXECUTIVE SUMMARY:")
        print("-" * 50)
        print(f"* Model achieved {metrics['accuracy']:.1%} accuracy on {total_samples} test samples")
        print(f"* {metrics['tp']} out of {metrics['tp'] + metrics['fn']} pneumonia cases correctly identified")
        print(f"* {metrics['tn']} out of {metrics['tn'] + metrics['fp']} normal cases correctly identified")
        print(f"* {metrics['fp']} false alarms, {metrics['fn']} missed cases")
        
        print(f"\nCONCLUSION: {performance} performance for medical AI application")
        print("="*80)

def main():
    """Main evaluation function"""
    
    print("Starting Model Evaluation")
    print("="*50)
    
    model_path = Path("../models/best_chest_xray_model.h5")
    data_path = Path("../data/processed")
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
        
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        return
    
    evaluator = ModelEvaluator(model_path, data_path)
    
    if not evaluator.load_model_and_data():
        print("Failed to load model or data")
        return
    
    predictions_prob, predictions_binary, true_labels = evaluator.generate_predictions()
    metrics = evaluator.calculate_all_metrics(predictions_prob, predictions_binary, true_labels)
    evaluator.print_comprehensive_report(metrics)
    
    # FIXED: Save metrics with proper type conversion
    results_path = Path("../results")
    results_path.mkdir(exist_ok=True)
    
    # All metrics are already converted to Python types in calculate_all_metrics
    with open(results_path / "evaluation_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nDetailed metrics saved successfully to: results/evaluation_metrics.json")
    
    # Sklearn classification report
    print(f"\nSKLEARN CLASSIFICATION REPORT:")
    print("-" * 50)
    report = classification_report(
        true_labels, 
        predictions_binary, 
        target_names=['Normal', 'Pneumonia'],
        digits=4
    )
    print(report)
    
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()
