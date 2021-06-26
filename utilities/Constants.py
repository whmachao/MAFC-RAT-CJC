import numpy as np

ROOT_DIR = '../'
DRAW_STATISTICAL_CHARACTERISTICS = False
SHOW_DETAILS = False
RECORD_KEYWORDS_AND_TFIDF = False

# Configuration for Train/Test Split ***********************************************************************************
TRAIN_TEST_SPLIT_RATIO = 0.8
SPLIT_BY_APP = True

# Configuration for Datasets *******************************************************************************************
SENSOR_DATASET_NAMES_INDICES = [6, 19, 28, 29, 36, 37, 39, 40, 47, 53, 54, 63, 64, 65, 72, 79]

SENSOR_DATASET_NAMES = ['Car', 'Earthquakes', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MoteStrain', 'Plane',
                        'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'Trace']


# Configuration for Hardware *******************************************************************************************

# if set 'USE_CPU' to True, CPU will be used for training deep learning classifiers.
# Otherwise, GPU will be used if available
USE_CPU = True
GPU_MEMO_ON_DEMAND = True

# Configuration for Pre-processing *************************************************************************************
MY_PREPROCESSORS = ['DPre', 'GANPre','DCGANPre']
GAN_SAVE_RESULTS_PATH = ""
GAN_AUGMENTATION_PERCENTAGE = 0.2
GAN_TRAINING_EPOCHS = 8000
GAN_TRAINING_BATCHSIZE = 32

# Configuration for Representation Learning ****************************************************************************
MY_REPRESENTATION_GENERATORS = ['PLAIN_HISTO', 'LEVEL_HISTO', 'MCT_MTS', 'MCT_SEMANTIC_MTS']

ENABLE_ECDF = True
WINDOW_SIZE = 2
WINDOW_STRIDE = 1
DATA_POINTS_LIST = range(10, 61, 10)

FIXED_LENGTH_LIST = [2**x for x in range(3, 5)]
TREE_LEVEL_LIST = range(1, 3)
GRAM_NUMBER_LIST = range(1, 3)

# Configuration for Extractor_Transformer_Loader ***********************************************************************
MY_ETL_COMPONENTS = ['PLAIN_HISTO', 'LEVEL_HISTO', 'MCT_MTS', 'MCT_SEMANTIC_MTS']

NORMALIZE_TIME_SERIES = True

# Following 3 parameters are used to filter out the methods according to their statistical characteristics
ENABLE_FILTERING = False
DURATION_PERCENTAGE = 0.001
VARIABLE_ACCESS_TIMES = 1
TOP_K_DURATION = 30

# Following ?? parameters are used to construct the semantic vectors for keywords
FIXED_INPUT_MATRIX_SIZE = 3000
TOP_K_KEYWORDS_LIST = range(10, 21, 10)
# TOP_K_KEYWORDS_LIST = range(10, 61, 10)

# Configuration for Classifiers ****************************************************************************************

MY_CLASSIFIERS = ['MLP', 'LSTM', 'FCN', 'ResNet','FCN_CAM', 'ResNet_CAM']

KNN_K = 1
KNN_STRATEGY = 'classmost'
KNN_DISTANCE = 'euclidean'

ITERATIONS = 5

# For UCR2015 archive, BATCH_SIZE is set to 128
# BATCH_SIZE = 128
# For wifi rate prediction dataset, BATCH_SIZE is set to 1024
# BATCH_SIZE = 1024
# For APP_TRACE_2019  archive, BATCH_SIZE is set to 32
BATCH_SIZE = 64

VERBOSE = 2

# if set 'ONLY_CSV_RESULTS' to True, for deep learning models, only save df_metrics.csv to minimize storage requirement
ONLY_CSV_RESULTS = False

EPOCHS = 20

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