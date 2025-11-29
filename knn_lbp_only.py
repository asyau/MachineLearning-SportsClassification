import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.neighbors import KNeighborsClassifier
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
import time
import json
from datetime import datetime
from skimage.feature import local_binary_pattern


class FeatureExtractor:
    """Extract LBP (Texture) features only"""

    def __init__(self, img_size=(224, 224), lbp_config=None):
        self.img_size = img_size

        # Default LBP configuration (Texture)
        if lbp_config is None:
            self.lbp_config = {
                'radius': 3,            # Radius of circle
                'n_points': 24,         # Number of points (usually 8 * radius)
                'method': 'uniform'     # Uniform patterns are rotation invariant
            }
        else:
            self.lbp_config = lbp_config

    def extract_lbp_features(self, image):
        """Extract Local Binary Patterns (LBP) features for texture analysis."""
        if isinstance(image, Image.Image):
            image = np.array(image)

        # LBP requires grayscale
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image

        # Extract LBP
        lbp = local_binary_pattern(
            gray_image,
            self.lbp_config['n_points'],
            self.lbp_config['radius'],
            method=self.lbp_config['method']
        )

        # Calculate the histogram of the LBP result
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

        # Normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)

        return hist

    def extract_features(self, image_path):
        """Extract LBP features from an image."""
        try:
            # Load and resize image
            image = Image.open(image_path).convert('RGB')
            image = image.resize(self.img_size)
            image_array = np.array(image)

            # Extract LBP features only
            lbp_features = self.extract_lbp_features(image_array)

            return lbp_features

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None


class KNNSportsClassifier:
    """K-Nearest Neighbors classifier for sports image classification"""

    def __init__(self, feature_extractor, use_pca=False, pca_components=None):
        self.feature_extractor = feature_extractor
        self.use_pca = use_pca
        self.pca_components = pca_components

        self.scaler = StandardScaler()
        self.pca = None
        self.knn_model = None
        self.class_names = None
        self.label_to_idx = None
        self.idx_to_label = None

    def load_dataset(self, data_dir, max_samples_per_class=None):
        features_list = []
        labels_list = []

        class_names = sorted([d for d in os.listdir(data_dir)
                             if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')])

        self.class_names = class_names
        self.label_to_idx = {label: idx for idx, label in enumerate(class_names)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        print(f"Found {len(class_names)} classes")
        print(f"Extracting LBP features from {data_dir}...")

        for class_name in tqdm(class_names, desc="Processing classes"):
            class_dir = os.path.join(data_dir, class_name)
            image_files = [f for f in os.listdir(class_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if max_samples_per_class:
                image_files = image_files[:max_samples_per_class]

            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                features = self.feature_extractor.extract_features(img_path)

                if features is not None:
                    features_list.append(features)
                    labels_list.append(self.label_to_idx[class_name])

        features = np.array(features_list)
        labels = np.array(labels_list)

        print(f"Extracted features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")

        return features, labels

    def train(self, train_dir, n_neighbors=5, metric='euclidean', weights='uniform',
              max_samples_per_class=None):
        print("\n" + "="*80)
        print("TRAINING K-NN CLASSIFIER (LBP ONLY)")
        print("="*80)

        start_time = time.time()

        # Load and extract features
        X_train, y_train = self.load_dataset(train_dir, max_samples_per_class)

        # Normalize features
        print("\nNormalizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Apply PCA
        if self.use_pca:
            print(f"\nApplying PCA...")
            if self.pca_components is None:
                self.pca = PCA(n_components=0.95, random_state=42)
            else:
                self.pca = PCA(n_components=self.pca_components, random_state=42)

            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            print(f"Reduced dimensions from {X_train.shape[1]} to {X_train_scaled.shape[1]}")
            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")

        # Train KNN
        print(f"\nTraining KNN with k={n_neighbors}, metric={metric}, weights={weights}...")
        self.knn_model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            weights=weights,
            n_jobs=-1
        )
        self.knn_model.fit(X_train_scaled, y_train)

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")

        return X_train_scaled, y_train

    def predict(self, test_dir, max_samples_per_class=None):
        print("\n" + "="*80)
        print("TESTING K-NN CLASSIFIER (LBP ONLY)")
        print("="*80)

        X_test, y_test = self.load_dataset(test_dir, max_samples_per_class)

        print("\nNormalizing features...")
        X_test_scaled = self.scaler.transform(X_test)

        if self.use_pca and self.pca is not None:
            X_test_scaled = self.pca.transform(X_test_scaled)

        print("Making predictions...")
        y_pred = self.knn_model.predict(X_test_scaled)

        return y_test, y_pred

    def evaluate(self, y_true, y_pred, save_dir='knn_results_lbp'):
        print("\n" + "="*80)
        print("EVALUATION RESULTS (LBP ONLY)")
        print("="*80)

        os.makedirs(save_dir, exist_ok=True)

        # Global Metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        print(f"\nGlobal Metrics:")
        print(f"  Accuracy:           {accuracy:.4f}")
        print(f"  Macro F1:           {f1_macro:.4f}")
        print(f"  Weighted F1:        {f1_weighted:.4f}")

        metrics = {
            'feature_type': 'LBP_only',
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_weighted': float(f1_weighted),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(save_dir, 'global_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        # Per-Class Report
        report_dict = classification_report(y_true, y_pred, target_names=self.class_names, 
                                          zero_division=0, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        csv_path = os.path.join(save_dir, 'per_class_metrics.csv')
        report_df.to_csv(csv_path)
        print(f"Per-class metrics saved to: {csv_path}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        cm_df.to_csv(os.path.join(save_dir, 'confusion_matrix.csv'))
        self._plot_confusion_matrix(cm, save_dir)

        return metrics

    def _plot_confusion_matrix(self, cm, save_dir):
        n_classes = len(self.class_names)
        figsize = max(10, n_classes * 0.8)
        
        plt.figure(figsize=(figsize, figsize))
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)

        sns.heatmap(cm_normalized, 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names,
                    cmap='Blues', 
                    fmt='.2f', 
                    square=True,
                    annot=False)

        plt.title('Confusion Matrix - LBP Only (Normalized)', fontsize=20)
        plt.xlabel('Predicted Label', fontsize=16)
        plt.ylabel('True Label', fontsize=16)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()

        cm_plot_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(cm_plot_path, dpi=150)
        plt.close()
        print(f"Confusion matrix plot saved to {cm_plot_path}")

    def save_model(self, save_path='knn_model_lbp.pkl'):
        model_data = {
            'knn_model': self.knn_model,
            'scaler': self.scaler,
            'pca': self.pca,
            'class_names': self.class_names,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'use_pca': self.use_pca
        }
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\nModel saved to {save_path}")


def k_only_hyperparameter_search(train_dir, val_dir, save_dir='knn_k_search_lbp'):
    print("\n" + "="*80)
    print("K-NN HYPERPARAMETER SEARCH (K ONLY) - LBP FEATURES")
    print("="*80)

    os.makedirs(save_dir, exist_ok=True)
    k_values = [1, 10, 50, 100]
    results = []

    print(f"Testing k values: {k_values}")

    for k in k_values:
        print(f"\nTesting k={k}...")
        try:
            feature_extractor = FeatureExtractor()
            classifier = KNNSportsClassifier(feature_extractor=feature_extractor, use_pca=True)

            start_time = time.time()
            classifier.train(train_dir, n_neighbors=k, metric='euclidean', weights='distance')
            train_time = time.time() - start_time

            y_val_true, y_val_pred = classifier.predict(val_dir)
            accuracy = accuracy_score(y_val_true, y_val_pred)
            f1 = f1_score(y_val_true, y_val_pred, average='macro', zero_division=0)

            print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

            results.append({
                'k': k, 'accuracy': accuracy, 'f1_score': f1,
                'metric': 'euclidean', 'weights': 'distance', 'use_pca': True,
                'train_time': train_time
            })

        except Exception as e:
            print(f"Error with k={k}: {e}")

    results_df = pd.DataFrame(results)
    results_path = os.path.join(save_dir, 'k_search_results.csv')
    results_df.to_csv(results_path, index=False)

    best_result = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"\nBest K found: {int(best_result['k'])} with accuracy: {best_result['accuracy']:.4f}")

    best_config = {
        'k': int(best_result['k']),
        'metric': 'euclidean',
        'weights': 'distance',
        'use_pca': True
    }
    
    with open(os.path.join(save_dir, 'best_k_config.json'), 'w') as f:
        json.dump(best_config, f)

    return results_df, best_config


def main():
    # Paths
    base_dir = 'MachineLearning-SportsClassification'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'valid')
    test_dir = os.path.join(base_dir, 'test')

    try:
        import skimage
    except ImportError:
        print("ERROR: scikit-image is required. Run: pip install scikit-image")
        return

    max_samples = None
    run_k_search = True

    if run_k_search:
        results_df, best_config = k_only_hyperparameter_search(train_dir, val_dir)
        
        print("\n" + "="*80)
        print("TRAINING FINAL MODEL WITH BEST CONFIG (LBP ONLY)")
        print("="*80)
        
        feature_extractor = FeatureExtractor()
        classifier = KNNSportsClassifier(feature_extractor=feature_extractor, use_pca=best_config['use_pca'])
        
        classifier.train(
            train_dir=train_dir,
            n_neighbors=best_config['k'],
            metric=best_config['metric'],
            weights=best_config['weights']
        )
    else:
        feature_extractor = FeatureExtractor()
        classifier = KNNSportsClassifier(feature_extractor=feature_extractor, use_pca=True)
        classifier.train(train_dir, n_neighbors=5, metric='euclidean', weights='distance', max_samples_per_class=max_samples)

    classifier.save_model('knn_sports_classifier_lbp.pkl')
    y_test_true, y_test_pred = classifier.predict(test_dir, max_samples_per_class=max_samples)
    classifier.evaluate(y_test_true, y_test_pred, save_dir='knn_results_lbp')


if __name__ == '__main__':
    main()
