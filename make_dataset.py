from utils import generate_train_test, generate_dataset_from_raw

import config as cfg

# generate_train_test(cfg.RAW_DATASET_DIR, cfg.RAW_DATASET_SAVE_DIR, split_ratio=cfg.TRAIN_TEST_SPLIT_RATIO)
generate_dataset_from_raw(cfg.RAW_DATASET_DIR, cfg.RAW_DATASET_SAVE_DIR)