DATASET_NAME = "synthetic-diversified-2"
if DATASET_NAME == "real":
    TRAIN_PATH = "/home/myeghiaz/Storage/SatClass-Real-0.125m-50px/train"
    TEST_PATH = "/home/myeghiaz/Storage/SatClass-Real-0.125m-50px/test"
elif DATASET_NAME == "synthetic":
    TRAIN_PATH = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px/train"
    TEST_PATH = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px/test"
elif DATASET_NAME == "synthetic-diversified-1":
    TRAIN_PATH = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px-diversified-1/train"
    TEST_PATH = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px-diversified-1/test"
elif DATASET_NAME == "synthetic-diversified-2":
    TRAIN_PATH = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px-diversified-2/train"
    TEST_PATH = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px-diversified-2/test"
elif DATASET_NAME == "stan-data":
    TRAIN_PATH = "/home/myeghiaz/Storage/SatClass-Real-0.125m-50px/train"
    TEST_PATH = "/home/myeghiaz/Storage/GSD:0.125m_sample-size:50_mean-sampling-freq:1"
else:
    raise NotImplementedError

BATCH_SIZE = 1024
MODEL_WEIGHTS = "saved_models/synthetic-diversified2-vgg/model_final.pth"
# MODEL_WEIGHTS = None
NUM_CLASSES = 1 # number of foreground classes
LR = 0.000005
N_EPOCHS = 5
TEST_SIZE = 0.2
VAL_FREQ = 2
OUTPUT_DIR = "output/"
RESULTS_DIR = "results/"
EVAL_ONLY = True
BRIGHTNESS_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MESHES_DIR = "/home/myeghiaz/Storage/GAN-vehicles-1000"
SYNTHETIC_SAVE_DIR = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px-diversified-2"
ATTACK_LR = 0.05
APPLY_TRAIN_TRANSFORMS = True
MODEL_NAME = 'vgg16'