"""
Balanced Dataset Creation Module

Creates balanced training, validation, and test datasets from imbalanced chest X-ray data
by sampling equal numbers from each class and splitting into appropriate proportions.
"""

import os
import shutil
import logging
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetBalancer:
    """
    Creates balanced datasets for binary classification tasks.
    
    Handles class imbalance by undersampling the majority class to match
    the minority class size, then creates stratified train/validation/test splits.
    """
    
    def __init__(self, source_path, processed_path, train_split=0.7, val_split=0.2, test_split=0.1):
        """
        Initialize the dataset balancer.
        
        Args:
            source_path (str): Path to source dataset directory
            processed_path (str): Path to output processed dataset directory  
            train_split (float): Proportion for training set (default: 0.7)
            val_split (float): Proportion for validation set (default: 0.2)
            test_split (float): Proportion for test set (default: 0.1)
        """
        self.source_path = Path(source_path)
        self.processed_path = Path(processed_path)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Validate split proportions
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test splits must sum to 1.0")
    
    def _create_directory_structure(self):
        """Create the output directory structure for balanced dataset."""
        splits = ['train', 'val', 'test']
        classes = ['normal', 'pneumonia']
        
        for split in splits:
            for cls in classes:
                (self.processed_path / split / cls).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created directory structure at {self.processed_path}")
    
    def _collect_images(self, class_name):
        """
        Collect all images for a specific class from the source directory.
        
        Args:
            class_name (str): Name of the class ('normal' or 'pneumonia')
            
        Returns:
            list: List of Path objects for images in the class
        """
        class_path = self.source_path / "train" / class_name
        
        # Try both lowercase and uppercase folder names
        if not class_path.exists():
            class_path = self.source_path / "train" / class_name.upper()
        
        if not class_path.exists():
            logger.warning(f"Class directory not found: {class_name}")
            return []
        
        # Collect image files with common extensions
        image_extensions = ["*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG"]
        images = []
        
        for extension in image_extensions:
            images.extend(list(class_path.glob(extension)))
        
        logger.info(f"Found {len(images)} images for class '{class_name}'")
        return images
    
    def _balance_classes(self, normal_images, pneumonia_images):
        """
        Balance classes by undersampling to minority class size.
        
        Args:
            normal_images (list): List of normal image paths
            pneumonia_images (list): List of pneumonia image paths
            
        Returns:
            tuple: (balanced_images, balanced_labels)
        """
        if len(normal_images) == 0 or len(pneumonia_images) == 0:
            raise ValueError("One or both classes have no images")
        
        min_count = min(len(normal_images), len(pneumonia_images))
        logger.info(f"Balancing dataset to {min_count} images per class")
        
        # Random sampling with fixed seed for reproducibility
        np.random.seed(42)
        selected_normal = np.random.choice(normal_images, min_count, replace=False)
        selected_pneumonia = np.random.choice(pneumonia_images, min_count, replace=False)
        
        # Combine selected images with labels
        all_images = list(selected_normal) + list(selected_pneumonia)
        all_labels = ['normal'] * min_count + ['pneumonia'] * min_count
        
        logger.info(f"Created balanced dataset with {len(all_images)} total images")
        return all_images, all_labels
    
    def _create_splits(self, images, labels):
        """
        Create stratified train/validation/test splits.
        
        Args:
            images (list): List of image paths
            labels (list): List of corresponding labels
            
        Returns:
            tuple: Six lists (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, 
            test_size=self.test_split, 
            stratify=labels, 
            random_state=42
        )
        
        # Second split: separate train/val from remaining data
        val_size_adjusted = self.val_split / (self.train_split + self.val_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size_adjusted, 
            stratify=y_temp, 
            random_state=42
        )
        
        logger.info(f"Created splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _copy_files(self, file_list, label_list, split_name):
        """
        Copy files to their respective directories in the processed dataset.
        
        Args:
            file_list (list): List of source file paths
            label_list (list): List of corresponding labels
            split_name (str): Name of the split ('train', 'val', or 'test')
        """
        copied_count = 0
        
        for file_path, label in zip(file_list, label_list):
            try:
                filename = file_path.name
                dest_dir = self.processed_path / split_name / label
                dest_path = dest_dir / filename
                
                shutil.copy2(file_path, dest_path)
                copied_count += 1
                
            except Exception as e:
                logger.error(f"Error copying {file_path}: {e}")
        
        logger.info(f"Copied {copied_count} files to {split_name} split")
    
    def _verify_balance(self):
        """
        Verify the final dataset balance and log statistics.
        
        Returns:
            dict: Dictionary containing dataset statistics
        """
        stats = {}
        total_files = 0
        
        for split in ['train', 'val', 'test']:
            normal_count = len(list((self.processed_path / split / 'normal').glob('*')))
            pneumonia_count = len(list((self.processed_path / split / 'pneumonia').glob('*')))
            split_total = normal_count + pneumonia_count
            total_files += split_total
            
            balance_ratio = normal_count / pneumonia_count if pneumonia_count > 0 else 0
            
            stats[split] = {
                'normal': normal_count,
                'pneumonia': pneumonia_count,
                'total': split_total,
                'balance_ratio': balance_ratio
            }
            
            logger.info(f"{split.upper()} - Normal: {normal_count}, Pneumonia: {pneumonia_count}, "
                       f"Total: {split_total}, Balance: {balance_ratio:.3f}")
        
        stats['total_files'] = total_files
        logger.info(f"Total balanced dataset: {total_files} files")
        
        return stats
    
    def create_balanced_dataset(self):
        """
        Execute the complete dataset balancing pipeline.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Starting dataset balancing pipeline")
            
            # Create output directory structure
            self._create_directory_structure()
            
            # Collect images for both classes
            normal_images = self._collect_images('normal')
            pneumonia_images = self._collect_images('pneumonia')
            
            if not normal_images or not pneumonia_images:
                logger.error("Failed to collect images for one or both classes")
                return False
            
            # Balance the classes
            balanced_images, balanced_labels = self._balance_classes(normal_images, pneumonia_images)
            
            # Create train/val/test splits
            X_train, X_val, X_test, y_train, y_val, y_test = self._create_splits(
                balanced_images, balanced_labels
            )
            
            # Copy files to processed structure
            self._copy_files(X_train, y_train, 'train')
            self._copy_files(X_val, y_val, 'val')
            self._copy_files(X_test, y_test, 'test')
            
            # Verify final balance
            stats = self._verify_balance()
            
            logger.info("Dataset balancing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Dataset balancing failed: {e}")
            return False

def create_balanced_dataset(source_path="../data/raw/chest_xray", 
                          processed_path="../data/processed",
                          train_split=0.7, val_split=0.2, test_split=0.1):
    """
    Convenience function to create balanced dataset with default parameters.
    
    Args:
        source_path (str): Path to source dataset directory
        processed_path (str): Path to output processed dataset directory
        train_split (float): Proportion for training set
        val_split (float): Proportion for validation set  
        test_split (float): Proportion for test set
        
    Returns:
        bool: True if successful, False otherwise
    """
    balancer = DatasetBalancer(source_path, processed_path, train_split, val_split, test_split)
    return balancer.create_balanced_dataset()

if __name__ == "__main__":
    success = create_balanced_dataset()
    
    if success:
        logger.info("Balanced dataset creation completed successfully")
    else:
