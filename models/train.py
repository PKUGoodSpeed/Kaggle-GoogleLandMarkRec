import os
import gc
import numpy as np
import pandas as pd
from skimage.io import imread
from keras.optimizers import SGD, Adam, Adadelta, RMSprop, Adagrad
from keras.callbacks import LearningRateScheduler, Callback, EarlyStopping, ModelCheckpoint

global_learning_rate = 0.01
global_decaying_rate = 0.92

def loadClassWeights(weights_file='../input/class_weights.json'):
    """ Loading class weight from a Json file """
    import json
    if not os.path.exists(weights_file):
        return None
    jfile = open(weights_file, "r")
    d = json.load(jfile)
    class_weights = dict([
        (int(key), val) for key, val in d.items()    
    ])
    return class_weights

class Trainer:
    model = None
    model_name = None
    
    def __init__(self, model, model_name, ):
        self.model = model
        self.model_name = model_name

    def trainingSetUp(self, train_path, train_df, valid_df, 
    batch_size=16, num_classes=1):
        assert os.path.exists(train_path)
        self.train_path = train_path
        self.train_df = train_df
        self.valid_df = valid_df
        self.batch_size = batch_size
        self.num_classes = num_classes

    def _train_gen(self):
        X = []
        Y = []
        while True:
            for idx, pred in zip(self.train_df.id.tolist(), self.train_df.landmark_id.tolist()):
                img_file = self.train_path + "/" + idx + ".jpg"
                assert os.path.exists(img_file)
                try:
                    img_ = imread(img_file)
                except:
                    continue
                assert img_.shape == (128, 128, 3), "Loading Error: wrong shape!"
                img_ = (img_/256.).astype(np.float32)
                cls_ = np.zeros(self.num_classes)
                cls_[int(pred)] = 1.
                X.append(img_)
                Y.append(cls_)

                if len(X) == self.batch_size:
                    if len(Y) == self.batch_size:
                        yield (np.array(X), np.array(Y))
                    del X
                    del Y
                    X = []
                    Y = []
                    gc.collect()
            if len(X) > 0:
                if len(Y) == len(X):
                    yield (np.array(X), np.array(Y))
                del X
                del Y
                X = []
                Y = []
                gc.collect()


    def _valid_gen(self):
        X = []
        Y = []
        while True:
            for idx, pred in zip(self.valid_df.id.tolist(), self.valid_df.landmark_id.tolist()):
                img_file = self.train_path + "/" + idx + ".jpg"
                try:    
                    img_ = imread(img_file)
                except:
                    continue
                assert img_.shape == (128, 128, 3), "Loading Error: wrong shape!"
                img_ = (img_/256.).astype(np.float32)
                cls_ = np.zeros(self.num_classes)
                cls_[int(pred)] = 1.
                X.append(img_)
                Y.append(cls_)

                if len(X) == self.batch_size:
                    if len(Y) == self.batch_size:
                        yield (np.array(X), np.array(Y))
                    del X
                    del Y
                    X = []
                    Y = []
                    gc.collect()
            if len(X) > 0:
                if len(Y) == len(X):
                    yield (np.array(X), np.array(Y))
                del X
                del Y
                X = []
                Y = []
                gc.collect()

    def _valid_data(self, max_length=16917):
        X = []
        Y = []
        for idx, pred in zip(self.valid_df.id.tolist(), self.valid_df.landmark_id.tolist()):
            img_file = self.train_path + "/" + idx + ".jpg"
            try:
                img_ = imread(img_file)
            except:
                continue
            assert img_.shape == (128, 128, 3), "Loading Error: wrong shape!"
            img_ = (img_/256.).astype(np.float32)
            cls_ = np.zeros(self.num_classes)
            cls_[int(pred)] = 1.
            X.append(img_)
            Y.append(cls_)

            if len(X) == max_length:
                return np.array(X), np.array(Y)
        return np.array(X), np.array(Y)

    def train(self, class_weights=None, learning_rate=0.02, decaying_rate=0.9, 
    epochs=10, resume=False):
        '''train the model'''
        # compile the model first
        self.model.compile(optimizer=Adam(0.005), loss='categorical_crossentropy', metrics=['accuracy'])
        checker_path = "./output/checkers"
        if not os.path.exists(checker_path):
            os.makedirs(checker_path)
        checker_file = checker_path + "/" + self.model_name + ".h5"

        if resume and os.path.exists(checker_file):
            print("Loading model from {FILE} ...".format(FILE=checker_file))
            try:
                self.model.load_weights(checker_file)
            except:
                print("WARNING: Loading model failed!")

        global global_learning_rate
        global global_decaying_rate
        ## Setting learning rate explicitly
        global_learning_rate = learning_rate
        global_decaying_rate = decaying_rate
        
        ## Adaptive learning rate changing
        def scheduler(epoch):
            global global_learning_rate
            global global_decaying_rate
            if epoch%2 == 0:
                global_learning_rate *= global_decaying_rate
                print("CURRENT LEARNING RATE = " + str(global_learning_rate))
            return global_learning_rate
        change_lr = LearningRateScheduler(scheduler)
        
        ## Set early stopper:
        earlystopper = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto')
        
        ## Set Check point
        checkpointer = ModelCheckpoint(filepath=checker_file, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

        train_steps = int((len(self.train_df)+self.batch_size-1)/self.batch_size)
        valid_steps = int((len(self.valid_df)+self.batch_size-1)/self.batch_size)

        history = self.model.fit_generator(self._train_gen(), steps_per_epoch=train_steps, epochs=epochs, verbose=1, 
        callbacks=[earlystopper, checkpointer, change_lr], validation_data=self._valid_gen(), validation_steps=valid_steps,
        class_weight=class_weights)
        return history
    
    def save(self):
        ''' saving the model '''
        checker_path = "./output/checkers"
        if not os.path.exists(checker_path):
            os.makedirs(checker_path)
        checker_file = checker_path + "/" + self.model_name + "_dump.h5"
        print("Saving the model into {FILE} ...".format(FILE=checker_file))
        self.model.save(checker_file)

    def load(self, model_file):
        print("Loading model from {FILE} ...".format(FILE=model_file))
        self.model.load_weights(model_file)
    
    def predict(self, test_df, test_path, load=True, block_size=4096):
        assert os.path.exists(test_path)
        print(" Making predictions ...")
        if load:
            checker_path = "./output/checkers"
            if not os.path.exists(checker_path):
                os.makedirs(checker_path)
            checker_file = checker_path + "/" + self.model_name + ".h5"
            print("Loading model from {FILE} ...".format(FILE=checker_file))
            try:
                self.model.load_weights(checker_file)
            except:
                print("WARNING: Loading model failed!")
        indices = []
        preds = []
        X = []
        for idx in test_df.id.tolist():
            img_file = test_path + "/" + idx + ".jpg"
            try:
                img_ = imread(img_file)
            except:
                print("WARNING: cannot loading image " + img_file)
            assert img_.shape == (128, 128, 3), "Loading Error: wrong shape!"
            img_ = (img_/256.).astype(np.float32)
            X.append(img_)
            indices.append(idx)
            if len(X) == block_size:
                pred_ = self.model.predict(np.array(X), batch_size=32, verbose=1)
                preds += list(pred_)
                del X
                X = []
                gc.collect()
        pred_ = self.model.predict(np.array(X), batch_size=32, verbose=1)
        pred_ = pred_.astype(np.float32)
        preds += list(pred_)
        del X
        X = []
        gc.collect()
        print len(preds), len(indices), len(preds[0])
        return pd.DataFrame({
            "id": indices,
            "preds": preds
        })

    def getModel(self):
        return self.model
