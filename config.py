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
elif DATASET_NAME == "synthetic-diversified-2":
    TRAIN_PATH = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px-diversified-2/train"
    TEST_PATH = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px-diversified-2/test"
elif DATASET_NAME == "stan-non-centered":
    TRAIN_PATH = "/home/myeghiaz/Storage/SatClass-Real-0.125m-50px/train"
    TEST_PATH = "/home/myeghiaz/Storage/GSD:0.125m_sample-size:50_mean-sampling-freq:1"
elif DATASET_NAME == "stan-with-shadows":
    TRAIN_PATH = "/home/myeghiaz/Storage/shadowless-shadow-dataset/with-shadows/train"
    TEST_PATH = "/home/myeghiaz/Storage/shadowless-shadow-dataset/with-shadows/test"
elif DATASET_NAME == "stan-without-shadows":
    TRAIN_PATH = "/home/myeghiaz/Storage/shadowless-shadow-dataset/without-shadows/train"
    TEST_PATH = "/home/myeghiaz/Storage/shadowless-shadow-dataset/without-shadows/test"
else:
    raise NotImplementedError

BATCH_SIZE = 1024
MODEL_WEIGHTS = "saved_models/synthetic-augmented-vgg/model_final.pth"
# MODEL_WEIGHTS = None
NUM_CLASSES = 1 # number of foreground classes
LR = 0.000005
N_EPOCHS = 5
TEST_SIZE = 0.2
VAL_FREQ = 2
OUTPUT_DIR = "output/"
RESULTS_DIR = "results/"
EVAL_ONLY = False
BRIGHTNESS_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MESHES_DIR = "/home/myeghiaz/Storage/GAN-vehicles"
SYNTHETIC_SAVE_DIR = "/home/myeghiaz/Storage/SatClass-Synthetic-0.125m-50px-proba"
ATTACK_LR = 0.05
APPLY_TRAIN_TRANSFORMS = False
MODEL_NAME = 'vgg16'
HEATMAP_NAME = "corr_heatmap"
VISUALIZE_HEATMAP_SAMPLES = True
ATTACKED_PARAMS = ['textures']
ADVERSARIAL_SAVE_DIR = "/home/myeghiaz/Storage/SatClass-Adversarial-0.125m-50px-proba"
NUM_ADV_IMGS = 1