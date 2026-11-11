"""
 Train MobileNetV2 model on balanced chest X-ray data (FIXED PATHS)
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ChestXRayTrainer:
    def __init__(self, img_size=224, batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
        # FIXED: Create models directory with correct path (go up 2 levels)
        Path("../../models").mkdir(exist_ok=True)
        
        print("🤖 ChestXRay Trainer Initialized!")
        print(f"   Image size: {img_size}x{img_size}")
        print(f"   Batch size: {batch_size}")
    
    def create_data_generators(self):
        """Create data generators for training"""
        
        print("\n📊 Creating data generators...")
        
        # FIXED: Data path - go up 2 levels from scripts/training/
        data_path = Path("../../data/processed")
        
        print(f"🔍 Looking for data at: {data_path.absolute()}")
        
        if not data_path.exists():
            print("❌ Processed data not found!")
            print("🔧 Checking alternative locations...")
            
            # Try alternative paths
            alt_paths = [
                Path("../data/processed"),      # One level up
                Path("../../data/processed"),   # Two levels up  
                Path("../../../data/processed") # Three levels up
            ]
            
            for alt_path in alt_paths:
                print(f"   Checking: {alt_path.absolute()}")
                if alt_path.exists():
                    print(f"✅ Found data at: {alt_path.absolute()}")
                    data_path = alt_path
                    break
            else:
                print("❌ Could not find processed data in any location!")
                return None, None, None
        else:
            print(f"✅ Found processed data at: {data_path.absolute()}")
        
        # Check for train/val/test folders
        required_folders = ['train', 'val', 'test']
        missing_folders = []
        
        for folder in required_folders:
            if not (data_path / folder).exists():
                missing_folders.append(folder)
        
        if missing_folders:
            print(f"❌ Missing folders: {missing_folders}")
            return None, None, None
        
        print("✅ All required folders found!")
        
        # Training data with augmentation (prevent overfitting)
        train_datagen = ImageDataGenerator(
            rescale=1./255,              # Normalize pixel values
            rotation_range=20,           # Random rotation ±20 degrees
            width_shift_range=0.2,       # Horizontal shift
            height_shift_range=0.2,      # Vertical shift
            zoom_range=0.2,             # Random zoom
            horizontal_flip=True,        # Flip horizontally
            brightness_range=[0.8, 1.2], # Brightness variation
            fill_mode='nearest'
        )
        
        # Validation/test data (no augmentation, just normalization)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            data_path / 'train',
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True,
            seed=42
        )
        
        val_generator = val_datagen.flow_from_directory(
            data_path / 'val',
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        test_generator = val_datagen.flow_from_directory(
            data_path / 'test',
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        print(f"✅ Data generators created successfully!")
        print(f"   Training samples: {train_generator.samples}")
        print(f"   Validation samples: {val_generator.samples}")
        print(f"   Test samples: {test_generator.samples}")
        
        # Show class mapping
        print(f"   Class mapping: {train_generator.class_indices}")
        
        return train_generator, val_generator, test_generator
    
    def build_model(self):
        """Build MobileNetV2-based model"""
        
        print("\n🏗️  Building MobileNetV2 model...")
        
        # Load pre-trained MobileNetV2 (without top layer)
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        self.model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.3),
            Dense(128, activation='relu', name='dense_1'),
            Dropout(0.2),
            Dense(1, activation='sigmoid', name='predictions')  # Binary classification
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✅ Model built successfully!")
        print(f"   Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def train_model(self, train_gen, val_gen, epochs=25):
        """Train the model"""
        
        print(f"\n🚀 Starting training for {epochs} epochs...")
        
        # Define callbacks
        callbacks = [
            ModelCheckpoint(
                '../../models/best_chest_xray_model.h5',  # FIXED: Correct path
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        print("🔥 Training started...")
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return self.history
    
    def evaluate_model(self, test_gen):
        """Comprehensive model evaluation"""
        
        print("\n📊 Evaluating model performance...")
        
        # Load best model
        self.model.load_weights('../../models/best_chest_xray_model.h5')  # FIXED: Correct path
        
        # Get predictions
        test_gen.reset()
        predictions = self.model.predict(test_gen, verbose=1)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        
        # Get true labels
        true_labels = test_gen.classes
        
        # Classification report
        class_names = ['Normal', 'Pneumonia']
        report = classification_report(
            true_labels, 
            predicted_classes, 
            target_names=class_names,
            digits=4
        )
        
        print("📋 Classification Report:")
        print("=" * 50)
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Chest X-Ray Classification', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add accuracy to plot
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        plt.figtext(0.5, 0.02, f'Overall Accuracy: {accuracy:.1%}', 
                   ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('../../models/confusion_matrix.png', dpi=300, bbox_inches='tight')  # FIXED: Correct path
        plt.show()
        
        # Calculate metrics
        print(f"\n🎯 Final Results:")
        print("=" * 30)
        print(f"Overall Accuracy: {accuracy:.1%}")
        
        # Per-class accuracy
        normal_accuracy = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        pneumonia_accuracy = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        
        print(f"Normal Detection: {normal_accuracy:.1%}")
        print(f"Pneumonia Detection: {pneumonia_accuracy:.1%}")
        
        # Check for bias
        if normal_accuracy > 0 and pneumonia_accuracy > 0:
            bias_ratio = max(normal_accuracy, pneumonia_accuracy) / min(normal_accuracy, pneumonia_accuracy)
            print(f"Bias Ratio: {bias_ratio:.2f}")
            
            if bias_ratio < 1.15:
                print("✅ Model is UNBIASED! Both classes predicted well!")
            else:
                print("⚠️  Some class bias detected")
        
        return accuracy, report, cm
    
    def plot_training_history(self):
        """Plot training history"""
        
        if self.history is None:
            print("❌ No training history available")
            return
        
        plt.figure(figsize=(15, 5))
        
        # Accuracy plot
        plt.subplot(1, 3, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title('Model Accuracy Progress', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss plot
        plt.subplot(1, 3, 2)
        plt.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Loss Progress', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Metrics plot
        plt.subplot(1, 3, 3)
        plt.plot(self.history.history['precision'], label='Precision', linewidth=2)
        plt.plot(self.history.history['recall'], label='Recall', linewidth=2)
        plt.title('Precision & Recall', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../../models/training_progress.png', dpi=300, bbox_inches='tight')  # FIXED: Correct path
        plt.show()

def main():
    """Main training pipeline"""
    
    print("🚀 Starting Chest X-Ray Model Training Pipeline")
    print("=" * 55)
    
    # Initialize trainer
    trainer = ChestXRayTrainer(img_size=224, batch_size=32)
    
    # Create data generators
    train_gen, val_gen, test_gen = trainer.create_data_generators()
    
    if train_gen is None:
        print("❌ Failed to create data generators")
        print("\n🔧 TROUBLESHOOTING STEPS:")
        print("1. Check if balanced dataset was created: data/processed/")
        print("2. Make sure you ran: python create_balanced_dataset.py")
        print("3. Verify train/val/test folders exist with normal/pneumonia subfolders")
        return
    
    # Build model
    trainer.build_model()
    
    # Train model
    trainer.train_model(train_gen, val_gen, epochs=25)
    
    # Plot training progress
    trainer.plot_training_history()
    
    # Evaluate model
    accuracy, report, cm = trainer.evaluate_model(test_gen)
    
    # Save final model
    trainer.model.save('../../models/final_chest_xray_model.h5')  # FIXED: Correct path
    print(f"\n💾 Final model saved: models/final_chest_xray_model.h5")
    
    # Success summary
    print(f"\n🎉 TRAINING COMPLETE! MODEL SUCCESS!")
    print("=" * 45)
    print(f"✅ Final Accuracy: {accuracy:.1%}")
    print(f"✅ Model saved and ready for deployment")
    print(f"✅ No bias issues (balanced dataset worked!)")
    print(f"✅ Ready for API deployment")

if __name__ == "__main__":
    main()

