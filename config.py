import os

"""Dataset parameters"""
# DATASET_NAMES = ['synthetic-diversified-1', 'adversarial-textures', 'adversarial-background', 'adversarial-shadows']
DATASET_NAMES = ['real']
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
if "adversarial-textures" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Adversarial-Textures-0.125m-50px")
if "adversarial-background" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Adversarial-Background-0.125m-50px")
if "adversarial-shadows" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Adversarial-Shadows-0.125m-50px")

"""Training parameters"""
NUM_GPUS = 2
if NUM_GPUS == 1:
    NUM_DATALOADER_WORKERS = 0
else:
    NUM_DATALOADER_WORKERS = 6
BATCH_SIZE = 1024
MODEL_WEIGHTS = None
# MODEL_WEIGHTS = "saved_models/non-centered-weighted-vgg/model_final.pt"
NUM_CLASSES = 1 # number of foreground classes
LR = 0.0000001
N_EPOCHS = 2
TEST_SIZE = 0.2
VAL_FREQ = 1
OUTPUT_DIR = "saved_models/non-centered-weighted-vgg/"
RESULTS_DIR = "results/"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
EVAL_ONLY = True
FP_FN_analysis = False
APPLY_TRAIN_TRANSFORMS = False
FOCAL_LOSS = {"alpha": 0.1, "gamma": 2}

"""Generating a train-test dataset from raw dataset"""
RAW_DATASET_DIR = "/var/storage/myeghiaz/GSD-0.125m_sample-size-50_mean-sampling-freq-1"
RAW_DATASET_SAVE_DIR = "/var/storage/myeghiaz/GSD-0.125m_sample-size-50"
TRAIN_TEST_SPLIT_RATIO = 0.0

"""Dataset generation parameters"""
SYNTHETIC_SAVE_DIR = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px-proba"
MESHES_DIR = "/home/myeghiaz/Storage/GAN-vehicles"
POSITIVE_LIMIT_TRAIN = None
NEGATIVE_LIMIT_TRAIN = None
POSITIVE_LIMIT_TEST = None
NEGATIVE_LIMIT_TEST = None

"""Attack parameters"""
BRIGHTNESS_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ATTACK_LR = 0.05
MODEL_NAME = 'vgg16'
HEATMAP_NAME = "corr_heatmap"
VISUALIZE_HEATMAP_SAMPLES = True
ATTACKED_PARAMS = ['shadows']
ADVERSARIAL_SAVE_DIR = "/home/myeghiaz/Storage/SatClass-Adversarial-0.125m-50px-proba"
NUM_ADV_IMGS = 10000 # Number of adversarial images that will be generated during the attack