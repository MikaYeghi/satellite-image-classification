DATASET_NAME = "real"
if DATASET_NAME == "real":
    TRAIN_PATH = "/home/myeghiaz/Storage/SatClass-Real-0.125m-50px/train"
    TEST_PATH = "/home/myeghiaz/Storage/SatClass-Real-0.125m-50px/test"
elif DATASET_NAME == "synthetic":
    TRAIN_PATH = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px/train"
    TEST_PATH = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px/test"
elif DATASET_NAME == "synthetic-diversified-1":
    TRAIN_PATH = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px-diversified-1/train"
    TEST_PATH = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px-diversified-1/test"
else:
    raise NotImplementedError

BATCH_SIZE = 1024
MODEL_WEIGHTS = "saved_models/synthetic-diversified-baseline/model_final.pth"
NUM_CLASSES = 1 # number of foreground classes
LR = 0.000005
N_EPOCHS = 5
TEST_SIZE = 0.2
VAL_FREQ = 1
OUTPUT_DIR = "output/"
RESULTS_DIR = "results/"
EVAL_ONLY = True
BRIGHTNESS_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MESHES_DIR = "/home/myeghiaz/Storage/GAN-vehicles"
SYNTHETIC_SAVE_DIR = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px-diversified-1"
ATTACK_LR = 0.05