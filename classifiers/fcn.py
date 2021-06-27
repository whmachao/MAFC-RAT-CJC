# FCN
import tensorflow.keras as keras
import numpy as np
import time
import utilities.Constants as Constants
from utilities.Utils import save_logs, delete_logs, get_optimal_batch_size


class Classifier_FCN:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=2):
        self.output_directory = output_directory
        self.model = self.build_model(input_shape, nb_classes)
        self.model.summary()
        print("")
        self.verbose = verbose
        self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        lr_patience = 1
        if Constants.EPOCHS * Constants.LR_PATIENCE_PERCENTAGE > 2:
            lr_patience = int(Constants.EPOCHS * Constants.LR_PATIENCE_PERCENTAGE)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=Constants.LR_MONITOR,
                                                      factor=Constants.LR_FACTOR,
                                                      patience=lr_patience,
                                                      verbose=Constants.LR_VERBOSE,
                                                      mode=Constants.LR_MODE,
                                                      min_delta=Constants.LR_MIN_DELTA,
                                                      cooldown=Constants.LR_COOLDOWN,
                                                      min_lr=Constants.LR_MIN)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true, batch, epochs, only_save_csv):
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = batch
        nb_epochs = epochs

        # mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))
        mini_batch_size = get_optimal_batch_size(x_train.shape[0], batch_size, 0.1)
        print('mini_batch_size is ' + str(mini_batch_size))

        start_training_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        end_training_time = time.time()
        training_time = end_training_time - start_training_time

        y_pred = self.model.predict(x_val)

        end_testing_time = time.time()
        testing_time = end_testing_time - end_training_time

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, training_time)

        if only_save_csv:
            delete_logs(self.output_directory)

        keras.backend.clear_session()

        return training_time, testing_time
