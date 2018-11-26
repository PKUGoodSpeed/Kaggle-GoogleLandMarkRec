import os
import gc
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

sys.path.append('./models')
sys.path.append('./utils')
from opts_parser import getopts
from train import Trainer, loadClassWeights
from naive import VggNet, Vgg19Net
from resnet import KerasResNet
from xception import XceptionNet
from inception import InceptionNet

TRAIN_PATH = './data/train'
VALID_PATH = './data/valid'
TEST_PATH = './data/test'
TRAIN_LABEL_FILE = './input/train_processed.csv'
VALID_LABEL_FILE = './input/valid_processed.csv'
TEST_LABEL_FILE = './input/test_processed.csv'
NUM_CLASSES = 14951
WEIGHTS_FILE = './input/class_weights.json'
INPUT_SHAPE = (128, 128, 3)

model_dict = {
    'vgg': VggNet,
    'kerasresnet': KerasResNet,
    'vgg19': Vgg19Net,
    'xception': XceptionNet,
    'inception': InceptionNet
}

if __name__ == '__main__':
    # getting configs
    C = getopts()
    print C

    # get model instance
    model_obj = model_dict[C['model_name']](input_shape=INPUT_SHAPE, output_dim=NUM_CLASSES)
    assert os.path.exists(TRAIN_LABEL_FILE), "Label file missing!"
    assert os.path.exists(TRAIN_PATH), "Training path does not exists"
    train_df = pd.read_csv(TRAIN_LABEL_FILE)
    valid_df = pd.read_csv(VALID_LABEL_FILE)
    print(train_df.shape)
    print(valid_df.shape)

    model_obj.buildModel(**C['model_kargs'])
    model = model_obj.getModel()
    trainer = Trainer(model=model, model_name=C['model_name'])
    trainer.trainingSetUp(TRAIN_PATH, train_df, valid_df, batch_size=64, num_classes=NUM_CLASSES)
    class_weights = loadClassWeights(WEIGHTS_FILE)
    trainer.train(resume=True, class_weights=class_weights, **C['train_args'])

    if not os.path.exists('./output'):
        os.makedirs('./output')
    
    oof_pred = trainer.predict(valid_df, TRAIN_PATH)

    oof_file = './output/' + C['model_name'] + "_oof.pik"
    print("Saving oof in {fname}".format(fname=oof_file))
    pickle.dump(oof_pred, open(oof_file, 'wb'))
    del train_df, valid_df, oof_pred
    gc.collect()

    assert os.path.exists(TEST_LABEL_FILE), "Label file missing!"
    assert os.path.exists(TEST_PATH), "Test path does not exists"

    test_df = pd.read_csv(TEST_LABEL_FILE)
    test_pred = trainer.predict(test_df, TEST_PATH)
    
    test_file = './output/' + C['model_name'] + "_test.pik"
    print("Saving test_predictions in {fname}".format(fname=test_file))
    pickle.dump(test_pred, open(test_file, 'wb'))
    del test_df, test_pred
    gc.collect()