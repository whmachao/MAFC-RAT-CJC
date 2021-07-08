import time
import numpy as np
import pandas as pd
from sklearn import svm

class Classifier_SVM:
    def __init__(self, output_directory, C=0.8, kernal='rbf', gamma=20, decision_function_shape='ovr'):
        self.output_directory = output_directory
        self.C = C
        self.kernal = kernal
        self.gamma = gamma
        self.decision_function_shape = decision_function_shape
        self.model = self.build_model()
        self.y_test_pred = list()

    def build_model(self):
        model = svm.SVC(C=self.C, kernel=self.kernal, gamma=self.gamma, decision_function_shape=self.decision_function_shape)
        return model

    def fit(self, x_train, y_train, x_test, y_test, y_true, batch=1, epochs=1, only_save_csv=1):
        start_time = time.time()
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
        y_train = self._get_true_labels(y_train)
        self.model.fit(x_train, y_train)
        training_time = time.time() - start_time

        start_time = time.time()
        train_accuracy = self.model.score(x_train, y_train)
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))
        y_test = self._get_true_labels(y_test)
        test_accuracy = self.model.score(x_test, y_test)
        testing_time = time.time() - start_time

        res = pd.DataFrame(columns=['best_model_train_acc', 'best_model_val_acc', 'time_consumption_in_seconds'])
        res.loc[0] = [train_accuracy, test_accuracy, training_time + testing_time]
        res.to_csv(path_or_buf=self.output_directory + 'df_metrics.csv', index=False)

        return training_time, testing_time

    def _get_true_labels(self, y_onehot):
        if not isinstance(y_onehot, np.ndarray) or len(y_onehot.shape) != 2:
            raise ValueError('y_onehot is not a 2-dimentional array!')

        y_true = []
        for i in range(y_onehot.shape[0]):
            y_true.append(np.argmax(y_onehot[i]) + 1)

        return y_true