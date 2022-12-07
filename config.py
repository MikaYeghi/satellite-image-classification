# TRAIN_PATH = "/home/myeghiaz/Storage/SatClass-Real-0.125m-50px/train"
# TEST_PATH = "/home/myeghiaz/Storage/SatClass-Real-0.125m-50px/test"
TRAIN_PATH = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px/train"
TEST_PATH = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px/test"
BATCH_SIZE = 1024
MODEL_WEIGHTS = "saved_models/synthetic-baseline/model_final.pth"
# MODEL_WEIGHTS = None
NUM_CLASSES = 1 # number of foreground classes
LR = 0.000005
N_EPOCHS = 10
TEST_SIZE = 0.2
VAL_FREQ = 1
OUTPUT_DIR = "saved_models/synthetic-baseline/"
EVAL_ONLY = True
BRIGHTNESS_LEVELS = [1.0]
MESHES_DIR = "/home/myeghiaz/Storage/GAN-vehicles"
SYNTHETIC_SAVE_DIR = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px"
ATTACK_LR = 0.05