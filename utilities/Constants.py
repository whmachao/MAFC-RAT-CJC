import os


DRAW_STATISTICAL_CHARACTERISTICS = False
SHOW_DETAILS = False
RECORD_KEYWORDS_AND_TFIDF = False


# Configuration for Train/Test Split ***********************************************************************************
TRAIN_TEST_SPLIT_RATIO = 0.8
SPLIT_BY_APP = True


# Configuration for Datasets *******************************************************************************************
ARCHIVE_NAMES = ['APP_TRACES_CJC']
APP_CATEGORY_NAMES = ['Memo', 'Calendar', 'Photography', 'Mixed', 'Test']


# Configuration for Hardware *******************************************************************************************
# if set 'USE_CPU' to True, CPU will be used for training deep learning classifiers.
# Otherwise, GPU will be used if available
USE_CPU = True
GPU_MEMO_ON_DEMAND = True


# Configuration for Representation Learning ****************************************************************************
MY_REPRESENTORS = ['PLAIN_HISTO', 'LEVEL_HISTO', 'MCT_MTS', 'MCT_SEMANTIC_MTS', 'FEATURE_FINDER']

ENABLE_ECDF = True
WINDOW_SIZE = 2
WINDOW_STRIDE = 1
DATA_POINTS_LIST = range(10, 11, 10)
# DATA_POINTS_LIST = range(10, 61, 10)

FIXED_LENGTH_LIST = [2**x for x in range(3, 5)]
TREE_LEVEL_LIST = range(1, 3)
GRAM_NUMBER_LIST = range(1, 3)


# Configuration for Extractor_Transformer_Loader ***********************************************************************
MY_ETL_COMPONENTS = ['PLAIN_HISTO', 'LEVEL_HISTO', 'MCT_MTS', 'MCT_SEMANTIC_MTS']

BIN_NUMBER_LIST = range(3, 5, 1)   # PLAIN_HISTO

NORMALIZE_TIME_SERIES = True

# Following 3 parameters are used to filter out the methods according to their statistical characteristics
ENABLE_FILTERING = False
DURATION_PERCENTAGE = 0.001
VARIABLE_ACCESS_TIMES = 1
TOP_K_DURATION = 30


# Following parameters are used to construct the semantic vectors for keywords
FIXED_INPUT_MATRIX_SIZE = 3000
TOP_K_KEYWORDS_LIST = range(10, 11, 10)
# TOP_K_KEYWORDS_LIST = range(10, 61, 10)


# Configuration for Classifiers ****************************************************************************************
MY_CLASSIFIERS = ['MLP', 'LSTM', 'FCN', 'ResNet', 'KNN']

KNN_K = 1
KNN_STRATEGY = 'classmost'
KNN_DISTANCE = 'euclidean'

ITERATIONS = 2


# For APP_TRACE_2019  archive, BATCH_SIZE is set to 16
BATCH_SIZE = 64

VERBOSE = 2

# if set 'ONLY_CSV_RESULTS' to True, for deep learning models, only save df_metrics.csv to minimize storage requirement
ONLY_CSV_RESULTS = False

EPOCHS = 200

# Configure the learning rate
LR_MONITOR = 'loss'
LR_FACTOR = 0.5
LR_PATIENCE_PERCENTAGE = 0.1
LR_VERBOSE = 2
LR_MODE = 'auto'
LR_MIN_DELTA = 0.0001
LR_COOLDOWN = 0
LR_MIN = 0.1


# Configuration for Optimal Representation Filtering *******************************************************************


# Configuration for Visualization **************************************************************************************
AXIS_SHOW_NAME_DICT = {'archive':'Archive', 'dataset':'Data Set', 'preprocessor':'Preprocessor', 'dimensionality':'d',
                       'representation_generator':'Representation Generator', 'classifier':'Classifier',
                       'iteration':'Iteration', 'representation_key':'Representation Key', 'bin_num':'Bin Number',
                       'split_ratio':'Split Ratio', 'tree_level':'Tree Level', 'best_model_train_loss':'Train Loss',
                       'best_model_val_loss':'Test Loss', 'best_model_train_acc':'Train Accuracy',
                       'best_model_val_acc':'Highest Identification Accuracy', 'best_model_learning_rate':'Learning Rate',
                       'best_model_nb_epoch':'Stop Epoch', 'time_consumption_in_seconds':'Time in Seconds'}

LEGEND_SHOW_NAME_DICT = {'MCT_MTS': 'SCR', 'MCT_SEMANTIC_MTS': 'SR'}


if __name__ == '__main__':
    print()