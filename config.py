import os

"""Dataset parameters"""
DATASET_NAMES = ['synthetic-diversified-1']
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
if "organic-camouflage-centered" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Organic-Camouflages-0.125m-50px/train")
    TEST_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Organic-Camouflages-0.125m-50px/test")
if "pixelated-camouflage-centered-8" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Pixelated-Camouflages-0.125m-50px-block-8/train")
    TEST_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Pixelated-Camouflages-0.125m-50px-block-8/test")
if "pixelated-camouflage-centered-16" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Pixelated-Camouflages-0.125m-50px-block-16/train")
    TEST_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Pixelated-Camouflages-0.125m-50px-block-16/test")
if "pixelated-camouflage-centered-32" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Pixelated-Camouflages-0.125m-50px-block-32/train")
    TEST_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Pixelated-Camouflages-0.125m-50px-block-32/test")
if "pixelated-camouflage-centered-64" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Pixelated-Camouflages-0.125m-50px-block-64/train")
    TEST_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Pixelated-Camouflages-0.125m-50px-block-64/test") 
if "pixelated-random-camouflage-centered-8" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Random-Pixelated-Camouflages-0.125m-50px-block-8/train")
    TEST_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Random-Pixelated-Camouflages-0.125m-50px-block-8/test")
if "pixelated-random-camouflage-centered-16" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Random-Pixelated-Camouflages-0.125m-50px-block-16/train")
    TEST_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Random-Pixelated-Camouflages-0.125m-50px-block-16/test")
if "pixelated-random-camouflage-centered-32" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Random-Pixelated-Camouflages-0.125m-50px-block-32/train")
    TEST_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Random-Pixelated-Camouflages-0.125m-50px-block-32/test")
if "pixelated-random-camouflage-centered-64" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Random-Pixelated-Camouflages-0.125m-50px-block-64/train")
    TEST_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Random-Pixelated-Camouflages-0.125m-50px-block-64/test")
if "pixelated-random-camouflage-centered-16-lightened" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Random-Pixelated-Camouflages-0.125m-50px-block-16-light-intensity:0.5-1.0/train")
    TEST_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Random-Pixelated-Camouflages-0.125m-50px-block-16-light-intensity:0.5-1.0/test")

"""Training and testing parameters"""
NUM_GPUS = 2
if NUM_GPUS == 1:
    NUM_DATALOADER_WORKERS = 0
else:
    NUM_DATALOADER_WORKERS = 6
BATCH_SIZE = 1024
# MODEL_WEIGHTS = None
MODEL_WEIGHTS = "saved_models/synthetic-augmented-vgg/model_final.pth"
NUM_CLASSES = 1 # number of foreground classes
LR = 0.0000001
N_EPOCHS = 2
TEST_SIZE = 0.2
VAL_FREQ = 1
OUTPUT_DIR = "output/"
RESULTS_DIR = "results/"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
EVAL_ONLY = True
FP_FN_analysis = True
APPLY_TRAIN_TRANSFORMS = False
MODEL_NAME = 'vgg16'
FOCAL_LOSS = {"alpha": 0.1, "gamma": 2}

"""Generating a train-test dataset from raw dataset"""
RAW_DATASET_DIR = "/var/storage/myeghiaz/GSD-0.125m_sample-size-50_mean-sampling-freq-1"
RAW_DATASET_SAVE_DIR = "/var/storage/myeghiaz/GSD-0.125m_sample-size-50"
TRAIN_TEST_SPLIT_RATIO = 0.0

"""Dataset generation parameters"""
SYNTHETIC_SAVE_DIR = "/home/myeghiaz/Storage/SatClass-Synthetic-Random-Pixelated-Camouflages-0.125m-50px-block-16-proba"
MESHES_DIR = "/home/myeghiaz/Storage/GAN-vehicles-1000"
POSITIVE_LIMIT_TRAIN = 100
NEGATIVE_LIMIT_TRAIN = 100
POSITIVE_LIMIT_TEST = 5000
NEGATIVE_LIMIT_TEST = 5000
CAMOUFLAGE_TEXTURES_PATH = "/var/storage/myeghiaz/pixelated-camouflages/block-64"
DESCRIPTIVE_COLORS_PATH = "/home/myeghiaz/Storage/descriptive-colors-real-train-fraction-0.1"
DRESS_CAMOUFLAGE = "random" # fixed - uses CAMOUFLAGE_TEXTURES_PATH, random - builds random camouflages from DESCRIPTIVE_COLORS_PATH, None - no camouflage augmentation is applied
PIXELATION_BLOCK_SIZE = 16
NUM_DESCRIPTIVE_COLORS = 10
COLORS_PER_CAMOUFLAGE = 4

"""Attack parameters"""
BRIGHTNESS_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ATTACK_LR = 0.05
HEATMAP_NAME = "corr_heatmap"
VISUALIZE_HEATMAP_SAMPLES = True
ATTACKED_PARAMS = ['shadows']
ADVERSARIAL_SAVE_DIR = "/home/myeghiaz/Storage/SatClass-Adversarial-0.125m-50px"
NUM_ADV_IMGS = 10000 # Number of adversarial images that will be generated during the attack