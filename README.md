# MachineLearning-SportsClassification

A machine learning project for classifying 100 different sports from images.

## Dataset

The dataset contains **14,492 images** across **100 sports categories**, organized into:

- **Training set**: 11,551 images
- **Validation set**: 1,405 images  
- **Test set**: 1,536 images

### Sports Categories (100 total)

The dataset includes diverse sports such as: air hockey, ampute football, archery, arm wrestling, axe throwing, balance beam, baseball, basketball, BMX, boxing, cricket, curling, fencing, figure skating, football, golf, gymnastics, hockey, judo, lacrosse, NASCAR racing, Olympic wrestling, pole vault, rock climbing, rugby, skiing, snowboarding, surfing, swimming, tennis, volleyball, weightlifting, and many more.

### Dataset Structure

```
MachineLearning-SportsClassification/
├── train/          # Training images (11,551 images)
│   ├── air hockey/
│   ├── archery/
│   ├── baseball/
│   └── ... (100 categories)
├── test/           # Test images (1,536 images)
│   └── ... (100 categories)
└── valid/          # Validation images (1,405 images)
    └── ... (100 categories)
```

## Dataset Download

**Note**: The image datasets are not included in this repository due to size constraints.

### Option 1: Google Drive (Recommended)
Download the dataset from: [Add your link here]

### Option 2: Other sources
[Add alternative download links]

## Installation

```bash
# Clone the repository
git clone https://github.com/asyau/MachineLearning-SportsClassification.git
cd MachineLearning-SportsClassification

# Install dependencies (after you add requirements.txt)
pip install -r requirements.txt

# Download and extract the dataset to the project root
# Ensure you have train/, test/, and valid/ directories
```

## Usage

[Add instructions for training and using your model here]

## Project Structure

```
MachineLearning-SportsClassification/
├── README.md
├── train/              # Training dataset (not in repo)
├── test/               # Test dataset (not in repo)
└── valid/              # Validation dataset (not in repo)
```

## Future Work

- [ ] Implement CNN model for classification
- [ ] Add training scripts
- [ ] Add evaluation metrics
- [ ] Create inference script
- [ ] Deploy model as web app

## License

[Add your license here]
