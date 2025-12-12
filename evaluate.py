"""
Evaluation script - CNN modülünü kullanarak değerlendirme
"""

from cnn.config import Config
from cnn.evaluate import evaluate

if __name__ == "__main__":
    config = Config()
    
    # Test set evaluation
    test_metrics = evaluate(config, split='test')
    
    # Validation set evaluation (opsiyonel)
    # valid_metrics = evaluate(config, split='valid')

