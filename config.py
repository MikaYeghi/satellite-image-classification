DATASET_NAMES = ["synthetic"]
TRAIN_PATH = []
TEST_PATH = []
if "real" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Real-0.125m-50px/train")
    TEST_PATH.append("/home/myeghiaz/Storage/SatClass-Real-0.125m-50px/test")
if "synthetic" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px/train")
    TEST_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px/test")
if "synthetic-diversified-1" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px-diversified-1/train")
    TEST_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px-diversified-1/test")
if "synthetic-diversified-2" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px-diversified-2/train")
    TEST_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px-diversified-2/test")
if "stan-non-centered" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/GSD-0.125m_sample-size-50/train")
    TEST_PATH.append("/home/myeghiaz/Storage/GSD-0.125m_sample-size-50/test")
if "stan-with-shadows" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/shadowless-shadow-dataset/with-shadows/train")
    TEST_PATH.append("/home/myeghiaz/Storage/shadowless-shadow-dataset/with-shadows/test")
if "stan-without-shadows" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/shadowless-shadow-dataset/without-shadows/train")
    TEST_PATH.append("/home/myeghiaz/Storage/shadowless-shadow-dataset/without-shadows/test")

"""Training parameters"""
BATCH_SIZE = 1024
MODEL_WEIGHTS = "saved_models/non-centered-baseline-vgg/model_final.pth"
# MODEL_WEIGHTS = None
NUM_CLASSES = 1 # number of foreground classes
LR = 0.000001
N_EPOCHS = 1
TEST_SIZE = 0.2
VAL_FREQ = 10
# OUTPUT_DIR = "saved_models/non-centered-weighted-vgg/"
OUTPUT_DIR = "output/"
RESULTS_DIR = "results/"
EVAL_ONLY = True
FP_FN_analysis = False
APPLY_TRAIN_TRANSFORMS = True
FOCAL_LOSS = {"alpha": 0.5, "gamma": 2}

"""Generating a train-test dataset from raw dataset"""
RAW_DATASET_DIR = "/var/storage/myeghiaz/GSD-0.125m_sample-size-50_mean-sampling-freq-1"
RAW_DATASET_SAVE_DIR = "/var/storage/myeghiaz/GSD-0.125m_sample-size-50"
TRAIN_TEST_SPLIT_RATIO = 0.0

"""Attack parameters"""
BRIGHTNESS_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MESHES_DIR = "/home/myeghiaz/Storage/GAN-vehicles-1000"
SYNTHETIC_SAVE_DIR = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px-proba"
ATTACK_LR = 0.05
MODEL_NAME = 'vgg16'
HEATMAP_NAME = "corr_heatmap"
VISUALIZE_HEATMAP_SAMPLES = True
ATTACKED_PARAMS = ['textures']
ADVERSARIAL_SAVE_DIR = "/home/myeghiaz/Storage/SatClass-Adversarial-0.125m-50px-proba"
NUM_ADV_IMGS = 1