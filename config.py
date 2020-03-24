import os

TRAIN_PATH = './dataset/stage2/stage2_train/'
TEST_PATH = './dataset/stage2/stage2_test/'


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3


train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

sizes_test = []