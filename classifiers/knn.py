import time
import numpy as np
import pandas as pd
import utilities.Constants as Constants
from classifiers.evaluation_metrics.overall_accuracy import Evaluation_Accuracy

class Classifier_KNN:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=2):
        self.output_directory = output_directory
        self.model = self.build_model(input_shape, nb_classes)
        self.y_test_pred = list()
        self.verbose = verbose

    def build_model(self, input_shape, nb_classes):
        model = None
        return model

    def fit(self, x_train, y_train, x_test, y_test, y_true, batch=1, epochs=1, only_save_csv=1):
        start_time = time.time()
        self.x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
        self.y_train = self._get_true_labels(y_train)
        self.x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))
        self.y_test = self._get_true_labels(y_test)
        self.k_value = Constants.KNN_K
        self.pred_strategy = Constants.KNN_STRATEGY
        self.distance_metric = Constants.KNN_DISTANCE
        self._get_predicted_labels_for_all()
        print(self.y_test_pred)
        accuracy = Evaluation_Accuracy(self.y_test_pred, self.y_test).get_evaluation_metric()
        duration = time.time() - start_time

        res = pd.DataFrame(columns=['best_model_val_acc', 'time_consumption_in_seconds'])
        res.loc[0] = [accuracy, duration]
        res.to_csv(path_or_buf=self.output_directory + 'df_metrics.csv', index=False)



    '''
    Given the distance list between the current test time series and all train time series by considering
    k train TS with the shortest distance to the test TS, following strategies are supported:
    1) classshortest: the label of the train TS in the same class with the shortest summed up distance to the test TS
    2) classmost: the label of the train TS in the same class with the most samples among all other classes
    '''
    def _get_predicted_labels_for_all(self):
        import datetime
        flag = 0
        totalNum = self.x_test.shape[0]
        for i in range(self.x_test.shape[0]):
            flag += 1
            if flag % 100 == 0:
                nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(nowTime)
                print(flag, ' has been processed of ', totalNum)
            self.y_test_pred.append(self._get_predicted_label_for_one(self.x_test[i]))

        return

    def _get_predicted_label_for_one(self, x_vec_test):
        predicted_label = None
        knn_distance_label_dict = {}

        # get the distance for each pair of vecs
        distance_list = []
        # shape[0] means the num of rows
        for i in range(self.x_train.shape[0]):
            if self.distance_metric == 'euclidean':
                from classifiers.distance_measures.euclidean import Distance_Euclidean
                # this step can be optimized by applying Symmetric Matrices
                # x_train[i]是单个对象,x_vec_test也是单个对象
                distance = Distance_Euclidean(self.x_train[i], x_vec_test).get_distance_between_two_vecs()
            elif self.distance_metric == 'wasserstein':
                from classifiers.distance_measures.wasserstein import Distance_Wasserstein
                distance = Distance_Wasserstein(self.x_train[i], x_vec_test).get_distance_between_two_vecs()
            elif self.distance_metric=='dtw':
                from classifiers.distance_measures.dtw import Distance_DTW
                [path,distance]=Distance_DTW(self.x_train[i], x_vec_test).get_distance_between_two_vecs()
            else:
                raise ValueError(self.distance_metric + ' is not implemented yet!')
            distance_list.append(distance)

        if self.pred_strategy == 'classshortest':
            min_Distance = min(distance_list)
            min_distance_index = distance_list.index(min_Distance)
            corresponding_label = self.y_train[min_distance_index]
            predicted_label = int(corresponding_label)
        elif self.pred_strategy == 'classmost':
            for i in range(self.k_value):
                current_distance = min(distance_list)
                similar_x_train_index = distance_list.index(current_distance)
                corresponding_label = self.y_train[similar_x_train_index]
                knn_distance_label_dict[str(current_distance)] = corresponding_label
                distance_list[similar_x_train_index] = np.inf

            knn_label_count_dict = {}
            for key, value in knn_distance_label_dict.items():
                if knn_label_count_dict.get(value) is None:
                    knn_label_count_dict[value] = 1
                else:
                    knn_label_count_dict[value] = knn_label_count_dict[value] + 1
            most_count = 0
            for key, value in knn_label_count_dict.items():
                if int(value) > most_count:
                    most_count = int(value)
                    predicted_label = key
        else:
            raise ValueError('Strategy: ' + self.pred_strategy + ' is not implemented yet!')

        return predicted_label


    def _get_true_labels(self, y_onehot):
        if not isinstance(y_onehot, np.ndarray) or len(y_onehot.shape) != 2:
            raise ValueError('y_onehot is not a 2-dimentional array!')

        y_true = []
        for i in range(y_onehot.shape[0]):
            y_true.append(np.argmax(y_onehot[i]) + 1)

        return y_true


if __name__ == '__main__':
    print('KNN classifier could be tested!')
