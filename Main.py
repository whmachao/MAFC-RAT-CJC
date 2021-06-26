from utilities.Utils import create_classifier, create_representation_generator
from utilities.Utils import create_directory
import sklearn
import pandas as pd
from Visualizer import Visualizer
import utilities.Constants as Constants
import tkinter as tk
import os
import time
import numpy as np


def fit_classifier(dataset_dict, classifier_name, output_directory):
    x_train = dataset_dict[0]
    y_train = dataset_dict[1]
    x_test = dataset_dict[2]
    y_test = dataset_dict[3]

    # store the original formats of x_train, y_train, x_test, y_test to facilitate cam-values collection
    original_x_train, original_y_train, original_x_test, original_y_test = x_train, y_train, x_test, y_test

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # save orignal y because later we will use binary
    y_true = y_test.astype(np.int64)
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]

    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)
    training_time, testing_time = classifier.fit(x_train, y_train, x_test, y_test, y_true,
                                                 Constants.BATCH_SIZE, Constants.EPOCHS, Constants.ONLY_CSV_RESULTS)

    return training_time, testing_time


# this function is used to launch an experiment on a single representation of specific dataset
def conduct_experiment(archive_name, dataset_name, classifier_name, itr, datasets_dict,
                       representation_method, representation_name):
    output_directory = os.getcwd() + '/detailed_results/' + classifier_name + '/' + archive_name + itr + '/' + \
                       dataset_name + '/' + representation_method + '/' + representation_name + '/'

    output_directory = create_directory(output_directory)

    print('Method: ', archive_name, dataset_name, classifier_name, itr)
    training_time, testing_time = None, None

    if output_directory is None:
        print('Already done')
    else:

        training_time, testing_time = fit_classifier(datasets_dict, classifier_name, output_directory)

        print('DONE')

        # the creation of this directory means
        create_directory(output_directory + '/DONE')

    return training_time, testing_time


def grid_search_main_loop():
    for category_name in target_datasets:
        for representor_name in target_representors:
            my_generator = create_representation_generator(representor_name, {}, category_name)
            my_representations_dict = my_generator.get_all_representations_dict()
            # STEP 4: train/validate the classifiers on all types of representations ***************************
            for my_representation_key in my_representations_dict.keys():
                my_datasets_dict = my_representations_dict[my_representation_key]
                for classifier_name in target_classifiers:
                    for current_itr in range(Constants.ITERATIONS):
                        itr = '_itr_' + str(current_itr + 1)
                        conduct_experiment(Constants.ARCHIVE_NAMES[0], category_name, classifier_name,
                                           itr, my_datasets_dict, representor_name, my_representation_key)


def aggregate_local_detailed_results(results_dir, column_names_dict):
    res = pd.DataFrame(data=np.zeros((0, 18), dtype=np.float), index=[],
                       columns=['archive', 'dataset', 'preprocessor', 'representation_generator',
                                'classifier', 'iteration', 'representation_key',
                                'bin_num', 'split_ratio', 'tree_level',
                                'best_model_train_loss', 'best_model_val_loss',
                                'best_model_train_acc', 'best_model_val_acc',
                                'best_model_learning_rate', 'best_model_nb_epoch',
                                'time_consumption_in_seconds', 'batch_size'])
    if not os.path.exists(results_dir):
        return res

    dirs = os.listdir(results_dir)
    if len(dirs) == 0:
        return res

    for dir in dirs:
        result_file_dir = results_dir + dir + '/' + 'df_metrics.csv'
        print('Aggregating the result from ' + str(result_file_dir))
        try:
            df_metrics = pd.read_csv(result_file_dir)
        except:
            continue
        df_metrics['archive'] = column_names_dict['archive']
        df_metrics['dataset'] = column_names_dict['dataset']
        df_metrics['preprocessor'] = column_names_dict['preprocessor']
        df_metrics['representation_generator'] = column_names_dict['representation_generator']
        df_metrics['classifier'] = column_names_dict['classifier']
        df_metrics['iteration'] = column_names_dict['iteration']
        df_metrics['representation_key'] = dir
        df_metrics['batch_size'] = Constants.BATCH_SIZE
        if column_names_dict['representation_generator'] == 'BDT':
            df_metrics['bin_num'] = dir.split('_')[0]
            df_metrics['split_ratio'] = dir.split('_')[1]
            df_metrics['tree_level'] = dir.split('_')[2]
        elif column_names_dict['representation_generator'] == 'LEVEL_HISTO':
            df_metrics['bin_num'] = dir.split('_')[0]
            df_metrics['tree_level'] = dir.split('_')[1]
            df_metrics['n_gram'] = dir.split('_')[2]
        else:
            df_metrics['bin_num'] = dir
            df_metrics['split_ratio'], df_metrics['tree_level'] = None, None

        res = pd.concat((res, df_metrics), axis=0, sort=False)

    return res


def aggregate_all_detailed_results(source_url, classifier_names, dataset_names, representation_generators):
    my_all_results = pd.DataFrame()
    for classifier_name in classifier_names:
        for iteration in range(Constants.ITERATIONS):
            for dataset_name in dataset_names:
                for representation_generator in representation_generators:
                    column_names_dict = {'archive': Constants.ARCHIVE_NAMES[2],
                                         'dataset': dataset_name,
                                         'representation_generator': representation_generator,
                                         'classifier': classifier_name, 'iteration': str(iteration + 1)}
                    local_results_url = source_url + '/' + classifier_name + '/' + Constants.ARCHIVE_NAMES[2] + \
                                        '_itr_' + str(iteration + 1) + '/' + dataset_name + '/' \
                                        + representation_generator + '/'
                    res = aggregate_local_detailed_results(local_results_url, column_names_dict)
                    my_all_results = my_all_results.append(res)
    return my_all_results


if __name__ == '__main__':
    if Constants.USE_CPU:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    command_string = 'run_tsc_experiments'
    # command_string = 'aggregate_tsc_results'
    # command_string = 'visualize_tsc_results'

    # command_string = 'etl_into_ucr_format'

    target_representors = Constants.MY_REPRESENTORS[2:3]
    target_classifiers = Constants.MY_CLASSIFIERS[0:1]
    target_datasets = Constants.APP_CATEGORY_NAMES[-1:]

    if command_string == 'run_tsc_experiments':
        grid_search_main_loop()
    elif command_string == 'aggregate_tsc_results':
        source_dir_url = os.getcwd() + '/detailed_results/'
        all_results = aggregate_all_detailed_results(source_dir_url, target_classifiers, target_datasets, target_representors)

        save_dir_url = os.getcwd() + '/aggregated_results/'
        create_directory(save_dir_url)
        time_stamp = str(time.localtime().tm_year) + '_' + str(time.localtime().tm_mon) + '_' + \
                     str(time.localtime().tm_mday) + '_' + str(time.localtime().tm_hour) + '_' + str(time.localtime().tm_min)
        all_results.to_csv(save_dir_url + 'all_dl_results_' + time_stamp + '.csv', index=False)
    elif command_string == 'visualize_tsc_results':
        root = tk.Tk()
        app = Visualizer(root)
        app.mainloop()
    elif command_string == 'etl_into_ucr_format':
        # try to transform the raw trace into UCR-like dataset format
        print('Not implemented yet!')
    else:
        raise ValueError("No such command!")
