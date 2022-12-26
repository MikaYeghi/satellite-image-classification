from utils import generate_train_test

import config as cfg

generate_train_test(cfg.RAW_DATASET_DIR, cfg.RAW_DATASET_SAVE_DIR, split_ratio=cfg.TRAIN_TEST_SPLIT_RATIO)