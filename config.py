import os
shape_code = "x1.3z1.3"
block_size = 8
non_centered_diversified_code = "circular-margin-synthetic"

"""Dataset parameters"""
# DATASET_NAMES = ['real']
# DATASET_NAMES = ['synthetic-NC-NM-diversified-1']
# DATASET_NAMES = ['non-centered-no-margin']
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
if "synthetic-non-centered-no-margin" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-non-centered-0.125m-50px/train")
    TEST_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-non-centered-0.125m-50px/test")
if "adversarial-textures" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Adversarial-Textures-0.125m-50px")
if "adversarial-background" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Adversarial-Background-0.125m-50px")
if "adversarial-shadows" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Adversarial-Shadows-0.125m-50px")
if "organic-camouflage-centered" in DATASET_NAMES:
    TRAIN_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Organic-Camouflages-0.125m-50px/train")
    TEST_PATH.append("/home/myeghiaz/Storage/SatClass-Synthetic-Organic-Camouflages-0.125m-50px/test")
if f"pixelated-camouflage-centered-{block_size}" in DATASET_NAMES:
    TRAIN_PATH.append(f"/home/myeghiaz/Storage/SatClass-Synthetic-Pixelated-Camouflages-0.125m-50px-block-{block_size}/train")
    TEST_PATH.append(f"/home/myeghiaz/Storage/SatClass-Synthetic-Pixelated-Camouflages-0.125m-50px-block-{block_size}/test")
if f"pixelated-random-camouflage-centered-{block_size}" in DATASET_NAMES:
    TRAIN_PATH.append(f"/home/myeghiaz/Storage/SatClass-Synthetic-Random-Pixelated-Camouflages-0.125m-50px-block-{block_size}/train")
    TEST_PATH.append(f"/home/myeghiaz/Storage/SatClass-Synthetic-Random-Pixelated-Camouflages-0.125m-50px-block-{block_size}/test")
if "synthetic-NC-NM-diversified-1" in DATASET_NAMES:
    TRAIN_PATH.append(f"/home/myeghiaz/Storage/SatClass-Synthetic-non-centered-0.125m-50px-diversified-1/{non_centered_diversified_code}/train")
    TEST_PATH.append(f"/home/myeghiaz/Storage/SatClass-Synthetic-non-centered-0.125m-50px-diversified-1/{non_centered_diversified_code}/test")

"""Training and testing parameters"""
NUM_GPUS = 2
if NUM_GPUS == 1:
    NUM_DATALOADER_WORKERS = 0
else:
    NUM_DATALOADER_WORKERS = 8
BATCH_SIZE = 1024
NUM_CLASSES = 1 # number of foreground classes
LR = 0.000005
N_EPOCHS = 5
TEST_SIZE = 0.2
VAL_FREQ = 10
OUTPUT_DIR = "output/"
RESULTS_DIR = "results/"
MODEL_WEIGHTS = None
# MODEL_WEIGHTS = os.path.join(OUTPUT_DIR, "model_final.pt")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
EVAL_ONLY = False
FP_FN_analysis = True
APPLY_TRAIN_TRANSFORMS = True
MODEL_NAME = 'vgg16'
FOCAL_LOSS = {"alpha": 0.3, "gamma": 2}
SHUFFLE = True

"""Generating a train-test dataset from raw dataset"""
RAW_DATASET_DIR = "/home/myeghiaz/Storage/GSD-0.125m_sample-size-50_mean-sampling-freq-1"
RAW_DATASET_SAVE_DIR = "/home/myeghiaz/Storage/SatClass-Synthetic-non-centered-0.125m-50px-diversified-1/circular-margin-real"
CIRCULAR_MARGIN = True
CIRCULAR_MARGIN_SIZE = 25
TRAIN_TEST_SPLIT_RATIO = 0.0

"""Dataset generation parameters"""
SYNTHETIC_SAVE_DIR = "/home/myeghiaz/Storage/SatClass-Synthetic-non-centered-0.125m-50px-diversified-1/proba"
MESHES_DIR = "/var/storage/myeghiaz/GAN-vehicles-1000"
TRAIN_MESHES_FRACTION = 0.8
POSITIVE_LIMIT_TRAIN = 513
NEGATIVE_LIMIT_TRAIN = 154919
POSITIVE_LIMIT_TEST = 62
NEGATIVE_LIMIT_TEST = 41445
CAMOUFLAGE_TEXTURES_PATH = None
DESCRIPTIVE_COLORS_PATH = "/home/myeghiaz/Storage/descriptive-colors-real-train-fraction-0.1/cluster_centers_10.pth"
DRESS_CAMOUFLAGE = None # fixed - uses CAMOUFLAGE_TEXTURES_PATH, random - builds random camouflages from DESCRIPTIVE_COLORS_PATH, 'organic' - load organic camouflages and replace their colors with colors from the dataset, None - no camouflage augmentation is applied
PIXELATION_BLOCK_SIZE = 16
NUM_DESCRIPTIVE_COLORS = 10
COLORS_PER_CAMOUFLAGE = 4
# NUMBER_OF_VEHICLES_PROBABILITY_DISTRIBUTION = [0.0, 8395/9800, 9580/9800, 9772/9800, 9797/9800, 1.0]
NUMBER_OF_VEHICLES_PROBABILITY_DISTRIBUTION = [0.0, 8395/9800, 9580/9800, 9772/9800, 1.0]

"""Attack parameters"""
BRIGHTNESS_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ATTACK_LR = 0.1
HEATMAP_NAME = "corr_heatmap"
VISUALIZE_HEATMAP_SAMPLES = False
ATTACKED_PARAMS = ['pixelated-textures']
ADVERSARIAL_SAVE_DIR = "/home/myeghiaz/Storage/UniText-0.125m-50px-synthetic-non-centered-TV-NPS"
ATTACKED_PIXELATED_TEXTURE_BLOCK_SIZE = 32
NUM_ADV_IMGS = 10 # Number of adversarial images that will be generated during the attack

"""Unified texture attack parameters"""
ATTACK_LOSS_FUNCTION = "classcore+TV+NPS"
ATTACK_LOSS_FUNCTION_PARAMETERS = {
    "TV-coefficient": 2.0,
    "classcore-coefficient": 0.001,
    "NPS-coefficient": 0.002,
    "GMM-coefficient": 0.001,
    "classcore": 0,
    "NPS-colors-path": "/var/storage/myeghiaz/UniText-0.125m-50px-centered/NPS/printable_colors.pth",
    "GMMLoss-directory": "/home/myeghiaz/Storage/UniText-0.125m-50px-non-centered/gmm-results"
}
ATTACK_BATCH_SIZE = 512
ATTACK_N_EPOCHS = 3
ATTACK_BASE_LR = 0.1
ATTACK_LR_GAMMA = 0.3
CENTERED_IMAGES_ATTACK = False
UNIFIED_TEXTURES_PATH = "/var/storage/myeghiaz/UniText-0.125m-50px-synthetic-non-centered-TV-NPS/unified_adversarial_textures.png"