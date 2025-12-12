"""
Training script - CNN modülünü kullanarak eğitim
"""

from cnn.config import Config
from cnn.train import train

if __name__ == "__main__":
    config = Config()
    model, history = train(config)

