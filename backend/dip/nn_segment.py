#!/usr/bin/env python3
#
# Author: Andrew Maule
# Objective: Simple dense layer neural net for training image segmentation.
#
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping
from segment import Segment


class NNSegment(Segment):
    def __init__(self, modelPath=None):
        '''
        Neural network segmentation class.  If a pretrained model is passed to a constructor, then it is load
        '''
        super().__init__()
        if modelPath:
            json_file = open('{}.json'.format(modelPath), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
            # load weights into new model
            self.model.load_weights("{}.h5".format(modelPath))
        else:
            self.model = None
        return


    def train(self, foreground, background, hidden_unit_size=10, seed=63342909):
        '''
        Train a simple neural network for image segmentation using an image representing 
        the foreground pixel values and one or more images representing the background 
        pixel values.
        '''
        self.model = models.Sequential()
        #Recast foreground image into a 3*num_pixels (channels) vector with label value = 1 for foreground
        #Recast background image into a 3*num_pixels (channels) vector with label value = 0 for background
        row,col,nchannel = foreground.shape
        foreground.resize((row*col,nchannel))
        foreground_categories = np.ones(row*col)
        row,col,nchannel = background.shape
        background.resize((row*col,nchannel))
        background_categories = np.zeros(row*col)
        np.random.seed(seed) #Set seed
        combined_samples = np.random.permutation(np.concatenate((foreground,background)))
        np.random.seed(seed) #Reset seed -- Needed to have category labels consistent with permuted input samples
        combined_sample_categories  = np.concatenate((foreground_categories,background_categories))
        self.model.add(layers.Dense(hidden_unit_size, activation='relu', input_shape=(3,)))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='rmsprop',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        #combined_samples_train = combined_samples[0:combined_samples.shape[0]]
        #combined_sample_categories_train  = combined_samples[0:combined_sample_categories.shape[0]]
        #combined_samples_test = combined_samples[combined_samples.shape[0]:]
        #combined_sample_categories_test= combined_samples[combined_sample_categories.shape[0]:]
        earlyStoppingC  = EarlyStopping(patience=3, verbose=1, restore_best_weights=True)
        #Recast background image into a 3*num_pixels (channels) vector with label value = 0 for background
        #Randomly mix and concatentate the foreground/background values and their corresponding labels
        model.add(layers.Dense(hidden_unit_size, activation='relu', input_shape=(3,)))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile()
        history = self.model.fit(   x=np.transpose(combined_samples), 
                                    y=np.transpose(combined_sample_categories), 
                                    epochs=20, 
                                    verbose=1, 
                                    callbacks=[earlyStoppingC], 
                                    validation_split=0.2, 
                                    shuffle=False, 
                                    use_multiprocessing=True)
        return(history)


    def predict(self, image):
        image = super().preprocess(image) #Handle global segment preprocessing on image before doing binary classification
        row,col,nchannel = image.shape
        image.reshape((row*col,nchannel));
        predictions = (self.model.predict(image) >= 0.5) #Cast to a boolean -- anything more than 50% is a foreground pixel
        predictions.reshape(row,col)
        predictions = super().postprocess(predictions)
        return(predictions)

    def export(self, path):
        model_json = self.model.to_json()
        with open("{}.json".format(path), "w") as json_file:
                json_file.write(model_json)
                json_file.close()
                # serialize weights to HDF5
                self.model.save_weights("{}.h5".format(path))
