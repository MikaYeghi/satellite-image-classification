import os
shape_code = "x1.3z1.3"

"""Dataset parameters"""
DATASET_NAMES = ['synthetic']
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
if "non-centered-no-margin" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Real-non-centered-0.125m-50px-no-margin/train")
    TEST_PATH.append("/home/myeghiaz/Storage/SatClass-Real-non-centered-0.125m-50px-no-margin/test")
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
if "organic-recolored-camouflage-centered" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Organic-Recolored-Camouflages-0.125m-50px/train")
    TEST_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Organic-Recolored-Camouflages-0.125m-50px/test")
if "modified-shapes" in DATASET_NAMES:
    TRAIN_PATH.append(f"/home/myeghiaz/Storage/modified-shape-datasets/SatClass-Synthetic-Modified-Shapes-0.125m-50px/{shape_code}/train")
    TEST_PATH.append(f"/home/myeghiaz/Storage/modified-shape-datasets/SatClass-Synthetic-Modified-Shapes-0.125m-50px/{shape_code}/test")
# TRAIN_PATH.append("/home/myeghiaz/Storage/organic-camouflages-dataset-style/Cam-1")

"""Training and testing parameters"""
NUM_GPUS = 2
if NUM_GPUS == 1:
    NUM_DATALOADER_WORKERS = 0
else:
    NUM_DATALOADER_WORKERS = 6
BATCH_SIZE = 1024
MODEL_WEIGHTS = None
# MODEL_WEIGHTS = "saved_models/synthetic-augmented-vgg/model_final.pth"
NUM_CLASSES = 1 # number of foreground classes
LR = 0.000001
N_EPOCHS = 5
TEST_SIZE = 0.2
VAL_FREQ = 1
# OUTPUT_DIR = f"/home/myeghiaz/Storage/modified-shape-datasets/results/{shape_code}"
OUTPUT_DIR = "saved_models/non-centered-no-margin-baseline/"
RESULTS_DIR = "results/"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
EVAL_ONLY = False
FP_FN_analysis = True
APPLY_TRAIN_TRANSFORMS = False
MODEL_NAME = 'vgg16'
FOCAL_LOSS = {"alpha": 0.1, "gamma": 2}
SHUFFLE = True

"""Generating a train-test dataset from raw dataset"""
RAW_DATASET_DIR = "/home/myeghiaz/Storage/GSD-0.125m_sample-size-50_mean-sampling-freq-1"
RAW_DATASET_SAVE_DIR = "/home/myeghiaz/Storage/SatClass-Real-non-centered-0.125m-50px-no-margin"
TRAIN_TEST_SPLIT_RATIO = 0.0

"""Dataset generation parameters"""
SYNTHETIC_SAVE_DIR = "/home/myeghiaz/Storage/SatClass-Synthetic-Organic-Recolored-Camouflages-0.125m-50px"
MESHES_DIR = "/var/storage/myeghiaz/GAN-vehicles"
# MESHES_DIR = f"/home/myeghiaz/Storage/modified-shape-datasets/datasets-xz/{shape_code}"
TRAIN_MESHES_FRACTION = 0.8
POSITIVE_LIMIT_TRAIN = 10
NEGATIVE_LIMIT_TRAIN = 10
POSITIVE_LIMIT_TEST = 1000
NEGATIVE_LIMIT_TEST = 1000
CAMOUFLAGE_TEXTURES_PATH = "/home/myeghiaz/Storage/organic-camouflages-dataset-style"
DESCRIPTIVE_COLORS_PATH = "/home/myeghiaz/Storage/descriptive-colors-real-train-fraction-0.1/cluster_centers_5.pth"
DRESS_CAMOUFLAGE = None # fixed - uses CAMOUFLAGE_TEXTURES_PATH, random - builds random camouflages from DESCRIPTIVE_COLORS_PATH, 'organic' - load organic camouflages and replace their colors with colors from the dataset, None - no camouflage augmentation is applied
PIXELATION_BLOCK_SIZE = 16
NUM_DESCRIPTIVE_COLORS = 10
COLORS_PER_CAMOUFLAGE = 4

"""Attack parameters"""
BRIGHTNESS_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ATTACK_LR = 0.1
HEATMAP_NAME = "corr_heatmap"
VISUALIZE_HEATMAP_SAMPLES = True
ATTACKED_PARAMS = ['discrete-textures']
ADVERSARIAL_SAVE_DIR = "/home/myeghiaz/Storage/SatClass-Adversarial-Discrete-Textures-0.125m-50px"
PIXELATED_TEXTURE_BLOCK_SIZE = 8 # Setting to 1moves from discrete attacks to ordinary texture attacks
NUM_ADV_IMGS = 1 # Number of adversarial images that will be generated during the attack