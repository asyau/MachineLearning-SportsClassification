import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from skimage.feature import hog, local_binary_pattern


# --- 1. UNIVERSAL FEATURE EXTRACTOR ---
class UniversalFeatureExtractor:
    def __init__(self, feature_type='color', img_size=(224, 224)):
        self.feature_type = feature_type
        self.img_size = img_size
        self.color_bins = 32
        self.hog_config = {'orientations': 9, 'pixels_per_cell': (16, 16), 'cells_per_block': (2, 2),
                           'channel_axis': -1}
        self.lbp_config = {'radius': 3, 'n_points': 24, 'method': 'uniform'}

    def extract(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize(self.img_size)
            image_array = np.array(image)

            if self.feature_type == 'color':
                return self._extract_color(image_array)
            elif self.feature_type == 'hog':
                return self._extract_hog(image_array)
            elif self.feature_type == 'lbp':
                return self._extract_lbp(image_array)
            else:
                return None
        except Exception as e:
            print(f"Error reading {image_path}: {e}")
            return None

    def _extract_color(self, image):
        hist_features = []
        for channel in range(3):
            hist = cv2.calcHist([image], [channel], None, [self.color_bins], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-7)
            hist_features.append(hist)
        return np.concatenate(hist_features)

    def _extract_hog(self, image):
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        return hog(gray_image, **self.hog_config, visualize=False, feature_vector=True)

    def _extract_lbp(self, image):
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        lbp = local_binary_pattern(gray_image, **self.lbp_config)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        return hist.astype("float") / (hist.sum() + 1e-7)


# --- 2. SVM CLASSIFIER ---
class SVMSportsClassifier:
    def __init__(self, feature_extractor, use_pca=False):
        self.feature_extractor = feature_extractor
        self.use_pca = use_pca
        self.scaler = StandardScaler()
        self.pca = None
        self.model = None
        self.class_names = None
        self.label_to_idx = None

    def load_dataset(self, data_dir, dataset_name="Data"):
        features_list = []
        labels_list = []

        if not os.path.exists(data_dir):
            print(f"[ERROR] Directory not found: {data_dir}")
            return np.array([]), np.array([])

        self.class_names = sorted([d for d in os.listdir(data_dir)
                                   if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')])
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}

        print(f"\n[{dataset_name}] Loading... ({len(self.class_names)} classes) from {data_dir}")

        for class_name in tqdm(self.class_names, desc=f"Scanning {dataset_name}"):
            class_dir = os.path.join(data_dir, class_name)
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

            for img_file in image_files:
                feat = self.feature_extractor.extract(os.path.join(class_dir, img_file))
                if feat is not None:
                    features_list.append(feat)
                    labels_list.append(self.label_to_idx[class_name])

        return np.array(features_list), np.array(labels_list)

    def train(self, train_dir, valid_dir):
        print("\n" + "=" * 40)
        print(f"SVM TRAINING (Train + Valid Split) - {self.feature_extractor.feature_type.upper()}")
        print("=" * 40)

        # 1. Load Data
        X_train, y_train = self.load_dataset(train_dir, "TRAIN")
        X_val, y_val = self.load_dataset(valid_dir, "VALID")

        if len(X_train) == 0 or len(X_val) == 0:
            print("[CRITICAL ERROR] Training or Validation data is empty. Check your paths.")
            sys.exit(1)

        # 2. Preprocessing
        print("\nPreprocessing (Scaler)...")
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        if self.use_pca:
            print("Applying PCA (95% variance)...")
            self.pca = PCA(n_components=0.95, random_state=42)
            X_train = self.pca.fit_transform(X_train)
            X_val = self.pca.transform(X_val)
            print(f"Dimensionality reduced to: {X_train.shape[1]} features.")

        # 3. Combine Data for PredefinedSplit
        X_combined = np.vstack((X_train, X_val))
        y_combined = np.concatenate((y_train, y_val))

        # -1: Train (exclude from validation), 0: Valid (use for validation)
        train_indices = np.full((len(X_train),), -1, dtype=int)
        val_indices = np.full((len(X_val),), 0, dtype=int)
        test_fold = np.concatenate((train_indices, val_indices))
        ps = PredefinedSplit(test_fold)

        # 4. Grid Search
        print(f"\nStarting Grid Search... (Total samples: {len(X_combined)})")

        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto'],
            'class_weight': ['balanced']
        }

        svc = SVC(probability=False)
        grid = GridSearchCV(svc, param_grid, cv=ps, n_jobs=-1, verbose=1)
        grid.fit(X_combined, y_combined)

        self.model = grid.best_estimator_

        print(f"\nSelected Best Parameters: {grid.best_params_}")
        print(f"Best Validation Score: {grid.best_score_:.4f}")

    def evaluate(self, test_dir, save_dir='svm_results'):
        print("\n" + "=" * 40)
        print("TEST SET EVALUATION")
        print("=" * 40)

        X_test, y_test = self.load_dataset(test_dir, "TEST")

        if len(X_test) == 0:
            print("[ERROR] Test data is empty.")
            return

        X_test = self.scaler.transform(X_test)
        if self.use_pca:
            X_test = self.pca.transform(X_test)

        y_pred = self.model.predict(X_test)

        os.makedirs(save_dir, exist_ok=True)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        print(f"\nAccuracy: {acc:.4f}")
        print(f"Macro F1: {f1:.4f}")
        print("\nClassification Report:\n")
        report = classification_report(y_test, y_pred, target_names=self.class_names, zero_division=0)
        print(report)

        # Save Report
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'SVM Confusion Matrix ({self.feature_extractor.feature_type})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        print(f"Confusion Matrix saved to {save_dir}/confusion_matrix.png")
        # plt.show() # Uncomment if you want to see the plot window


# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    # ==========================================
    # CONFIGURATION SECTION - EDIT THIS PART
    # ==========================================

    # PLEASE CHANGE THE PATH BELOW TO YOUR DATASET DIRECTORY
    BASE_DIR = "YOUR_DATASET_DIRECTORY_HERE"

    # Directory where outputs (model and plots) will be saved
    OUTPUT_DIR = "svm_output"

    # Settings
    SELECTED_FEATURE = 'color'  # Options: 'color', 'hog', 'lbp'
    USE_PCA = True

    # ==========================================
    # END OF CONFIGURATION
    # ==========================================

    # Path check
    if BASE_DIR == "YOUR_DATASET_DIRECTORY_HERE":
        print("\n[WARNING] Please change the 'BASE_DIR' variable to your own dataset path!")
        print("Example: BASE_DIR = '/Users/ali/Desktop/SportsDataset'")
        sys.exit(1)

    if not os.path.exists(BASE_DIR):
        print(f"\n[ERROR] Specified directory not found: {BASE_DIR}")
        sys.exit(1)

    train_path = os.path.join(BASE_DIR, 'train')
    valid_path = os.path.join(BASE_DIR, 'valid')
    test_path = os.path.join(BASE_DIR, 'test')

    # Folder structure check
    if not (os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path)):
        print(f"\n[ERROR] '{BASE_DIR}' must contain 'train', 'valid', and 'test' folders.")
        sys.exit(1)

    print(f"--- Process Started: {SELECTED_FEATURE.upper()} features ---")

    # 1. Initialize
    extractor = UniversalFeatureExtractor(feature_type=SELECTED_FEATURE)
    classifier = SVMSportsClassifier(feature_extractor=extractor, use_pca=USE_PCA)

    # 2. Train
    classifier.train(train_path, valid_path)

    # 3. Save Model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_save_path = os.path.join(OUTPUT_DIR, f'svm_model_{SELECTED_FEATURE}.pkl')

    save_data = {
        'model': classifier.model,
        'scaler': classifier.scaler,
        'pca': classifier.pca,
        'class_names': classifier.class_names
    }
    with open(model_save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nModel saved to: {model_save_path}")

    # 4. Evaluate
    classifier.evaluate(test_path, save_dir=os.path.join(OUTPUT_DIR, 'results'))

    print("\n--- Process Completed Successfully ---")