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

# Import scikit-image features

from skimage.feature import hog, local_binary_pattern





class FeatureExtractor:

    """Extract hand-crafted features: Color Histograms + HOG (Shape) + LBP (Texture)"""



    def __init__(self, img_size=(224, 224), color_bins=32, hog_config=None, lbp_config=None):

        """

        Initialize feature extractor.

        Args:

            img_size: Tuple (height, width) for resizing images

            color_bins: Number of bins per color channel for histogram

            hog_config: Dictionary with HOG parameters

            lbp_config: Dictionary with LBP parameters

        """

        self.img_size = img_size

        self.color_bins = color_bins



        # Default HOG configuration (Shape)

        if hog_config is None:

            self.hog_config = {

                'orientations': 9,

                'pixels_per_cell': (16, 16),

                'cells_per_block': (2, 2),

                'channel_axis': -1

            }

        else:

            self.hog_config = hog_config



        # Default LBP configuration (Texture)

        if lbp_config is None:

            self.lbp_config = {

                'radius': 3,            # Radius of circle

                'n_points': 24,         # Number of points (usually 8 * radius)

                'method': 'uniform'     # Uniform patterns are rotation invariant

            }

        else:

            self.lbp_config = lbp_config



    def extract_color_histogram(self, image):

        """Extract color histogram features from RGB image."""

        if isinstance(image, Image.Image):

            image = np.array(image)



        # Ensure image is RGB

        if len(image.shape) == 2:  # Grayscale

            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        elif image.shape[2] == 4:  # RGBA

            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)



        # Compute histogram for each channel

        hist_features = []

        for channel in range(3):

            hist = cv2.calcHist([image], [channel], None, [self.color_bins], [0, 256])

            hist = hist.flatten()

            # Normalize histogram

            hist = hist / (hist.sum() + 1e-7)

            hist_features.append(hist)



        return np.concatenate(hist_features)



    def extract_hog_features(self, image):

        """Extract Histogram of Oriented Gradients (HOG) features."""

        if isinstance(image, Image.Image):

            image = np.array(image)



        # Convert to grayscale for HOG

        if len(image.shape) == 3:

            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        else:

            gray_image = image



        # Extract HOG features

        hog_features = hog(

            gray_image,

            orientations=self.hog_config['orientations'],

            pixels_per_cell=self.hog_config['pixels_per_cell'],

            cells_per_block=self.hog_config['cells_per_block'],

            visualize=False,

            feature_vector=True

        )



        return hog_features



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

        """

        Extract combined features (Color + HOG + LBP) from an image.

        This is where concatenation happens.

        """

        try:

            # Load and resize image

            image = Image.open(image_path).convert('RGB')

            image = image.resize(self.img_size)

            image_array = np.array(image)



            # 1. Extract color histogram

            color_hist = self.extract_color_histogram(image_array)



            # 2. Extract HOG features (Shape)

            hog_features = self.extract_hog_features(image_array)



            # 3. Extract LBP features (Texture)

            lbp_features = self.extract_lbp_features(image_array)



            # Concatenate all features into one vector

            combined_features = np.concatenate([color_hist, hog_features, lbp_features])



            return combined_features



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

        print(f"Extracting features (Color + HOG + LBP) from {data_dir}...")



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

        print("TRAINING K-NN CLASSIFIER")

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

        print("TESTING K-NN CLASSIFIER")

        print("="*80)



        X_test, y_test = self.load_dataset(test_dir, max_samples_per_class)



        print("\nNormalizing features...")

        X_test_scaled = self.scaler.transform(X_test)



        if self.use_pca and self.pca is not None:

            X_test_scaled = self.pca.transform(X_test_scaled)



        print("Making predictions...")

        y_pred = self.knn_model.predict(X_test_scaled)



        return y_test, y_pred



    def evaluate(self, y_true, y_pred, save_dir='knn_results'):

        """

        Evaluate and save detailed per-class metrics and confusion matrix.

        """

        print("\n" + "="*80)

        print("EVALUATION RESULTS")

        print("="*80)



        os.makedirs(save_dir, exist_ok=True)



        # --- 1. Global Metrics ---

        accuracy = accuracy_score(y_true, y_pred)

        

        # Macro: Simple average across classes

        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)

        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

        

        # Weighted: Weighted by number of samples per class

        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)

        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)



        print(f"\nGlobal Metrics:")

        print(f"  Accuracy:           {accuracy:.4f}")

        print(f"  Macro F1:           {f1_macro:.4f}")

        print(f"  Weighted F1:        {f1_weighted:.4f}")



        # Save global metrics

        metrics = {

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



        # --- 2. Per-Class Detailed Report (CSV) ---

        print("\nGenerating per-class metrics...")

        

        # Get dictionary of metrics

        report_dict = classification_report(y_true, y_pred, target_names=self.class_names, 

                                          zero_division=0, output_dict=True)

        

        # Convert to DataFrame

        report_df = pd.DataFrame(report_dict).transpose()

        

        # Save to CSV

        csv_path = os.path.join(save_dir, 'per_class_metrics.csv')

        report_df.to_csv(csv_path)

        print(f"Per-class metrics saved to: {csv_path}")

        

        # Print text report to console

        print("\nClassification Report (Text):")

        print(classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0))



        # --- 3. Confusion Matrix ---

        cm = confusion_matrix(y_true, y_pred)

        

        # Save Raw Numbers

        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)

        cm_df.to_csv(os.path.join(save_dir, 'confusion_matrix.csv'))

        

        # Plot Heatmap

        self._plot_confusion_matrix(cm, save_dir)



        return metrics



    def _plot_confusion_matrix(self, cm, save_dir):

        """Plot and save confusion matrix visualization."""

        # Calculate figure size based on number of classes

        n_classes = len(self.class_names)

        figsize = max(10, n_classes * 0.8) # Dynamic sizing

        

        plt.figure(figsize=(figsize, figsize))

        

        # Normalize for color mapping (better visualization)

        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)



        sns.heatmap(cm_normalized, 

                    xticklabels=self.class_names, 

                    yticklabels=self.class_names,

                    cmap='Blues', 

                    fmt='.2f', 

                    square=True,

                    annot=False) # Turn off annotation if too many classes



        plt.title('Confusion Matrix (Normalized)', fontsize=20)

        plt.xlabel('Predicted Label', fontsize=16)

        plt.ylabel('True Label', fontsize=16)

        plt.xticks(rotation=90)

        plt.yticks(rotation=0)

        plt.tight_layout()



        cm_plot_path = os.path.join(save_dir, 'confusion_matrix.png')

        plt.savefig(cm_plot_path, dpi=150)

        plt.close()

        print(f"Confusion matrix plot saved to {cm_plot_path}")



    def save_model(self, save_path='knn_model.pkl'):

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





def k_only_hyperparameter_search(train_dir, val_dir, save_dir='knn_k_search'):

    """Simplified hyperparameter search for K values only"""

    print("\n" + "="*80)

    print("K-NN HYPERPARAMETER SEARCH (K ONLY)")

    print("="*80)



    os.makedirs(save_dir, exist_ok=True)

    k_values = [1, 10, 50, 100]

    results = []



    print(f"Testing k values: {k_values}")



    for k in k_values:

        print(f"\nTesting k={k}...")

        try:

            # Recreate classifier for fresh start

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



    # Ensure skimage is installed

    try:

        import skimage

    except ImportError:

        print("ERROR: scikit-image is required. Run: pip install scikit-image")

        return



    # Configuration

    max_samples = None  # Set to 10 for quick debugging, None for full run

    run_k_search = True # Set to True to optimize K



    if run_k_search:

        results_df, best_config = k_only_hyperparameter_search(train_dir, val_dir)

        

        print("\n" + "="*80)

        print("TRAINING FINAL MODEL WITH BEST CONFIG")

        print("="*80)

        

        feature_extractor = FeatureExtractor() # Includes Color + HOG + LBP

        classifier = KNNSportsClassifier(feature_extractor=feature_extractor, use_pca=best_config['use_pca'])

        

        classifier.train(

            train_dir=train_dir,

            n_neighbors=best_config['k'],

            metric=best_config['metric'],

            weights=best_config['weights']

        )

    else:

        # Default run without search

        feature_extractor = FeatureExtractor()

        classifier = KNNSportsClassifier(feature_extractor=feature_extractor, use_pca=True)

        classifier.train(train_dir, n_neighbors=5, metric='euclidean', weights='distance', max_samples_per_class=max_samples)



    # Final Evaluation

    classifier.save_model('knn_sports_classifier.pkl')

    y_test_true, y_test_pred = classifier.predict(test_dir, max_samples_per_class=max_samples)

    classifier.evaluate(y_test_true, y_test_pred)





if __name__ == '__main__':

    main()
