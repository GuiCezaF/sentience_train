from pathlib import Path
import torch

BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
ONNX_PATH = MODELS_DIR / "emotion_model.onnx"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
TRAINING_LOG_PATH = LOGS_DIR / "training.log"

MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 48
CHANNELS = 1
NUM_CLASSES = 7
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 80
PATIENCE = 12
K_FOLDS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
