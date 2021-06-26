import numpy as np
import time
import os
import AFD.AFD_Constants as Local_Constants
import utilities.Constants as Constants
from AFD.AFD_Utils import construct_method_call_tree, draw_2D_figure
from sklearn import manifold
import pandas as pd
from matplotlib.figure import Figure
from AFD.mts_representation_generator import MTS_Representation_Generator
from AFD.semantic_mts_representation_generator import Semantic_MTS_Representation_Generator


# Load and transform the raw traces of the specific dataset (i.e. category) into method call trees.
def load_transform_traces(dataset_name):
    print('Start to parse: ' + dataset_name)
    start_time = time.time()
    app_trace_dict = {}
    mct_depth_list, mct_nodes_num_list = list(), list()
    all_category_dir = Local_Constants.ROOT_DIR + 'archives' + '/' + Constants.ARCHIVE_NAMES[2]
    category_dir = all_category_dir + '/' + dataset_name + '/'
    for app_name in os.listdir(category_dir):
        trace_names_list = os.listdir(category_dir + '/' + app_name)
        for trace_name in trace_names_list:
            my_trace_url = category_dir + '/' + app_name + '/' + trace_name
            my_trace_name = dataset_name + '&' + app_name + '&' + trace_name
            call_method_tree, exceptional_lines = construct_method_call_tree(my_trace_name, my_trace_url)
            app_trace_dict[my_trace_name] = call_method_tree
            if Local_Constants.SHOW_DETAILS:
                call_method_tree.show()
            print(my_trace_name + ' method call tree statistics:')
            print('Depth: ' + str(call_method_tree.depth()))
            print('No. of Nodes: ' + str(len(call_method_tree.all_nodes())))
            print('Lines not covered: ' + str(exceptional_lines))
            mct_depth_list.append(call_method_tree.depth())
            mct_nodes_num_list.append(len(call_method_tree.all_nodes()))
            print('---------------------------------------------------------------------------------')
    complete_time = time.time()
    print('Complete to parse: ' + dataset_name + ' using ' + str(complete_time - start_time) + ' seconds!')
    return app_trace_dict


# Draw the MCT depth (x-axis) vs. number of nodes (y-axis) 2D chart
def draw_2d_mct_depth_nodes(dataset_name, trace_mct_list):
    mct_depth_list, mct_nodes_num_list = list(), list()
    for call_method_tree in trace_mct_list:
        mct_depth_list.append(call_method_tree.depth())
        mct_nodes_num_list.append(len(call_method_tree.all_nodes()))
    my_title = dataset_name
    my_data_dict = dict()
    my_data_dict['x_label'], my_data_dict['y_label'] = 'MCT Depth', 'MCT Nodes'
    my_data_dict['x_values'], my_data_dict['y_values'] = mct_depth_list, mct_nodes_num_list
    draw_2D_figure(my_title, my_data_dict)
    return


# Reduce dimensionality of the given representation using t-SNE
# And show samples in 2D or 3D chart
def reduce_dim_and_show(dataset_name, rep_method, representation):
    # Vertically stack train/test samples and conduct dimensionality reduction to 2D by using t-SNE
    train_samples, train_labels = representation[0], representation[1].tolist()
    test_samples, test_labels = representation[2], representation[3].tolist()
    complete_samples_arr = np.vstack((train_samples, test_samples))
    start_time = time.time()
    tsne = manifold.TSNE(n_components=2, init='pca', n_iter_without_progress=500, n_iter=5000, method='exact',
                         learning_rate=300.0, perplexity=30.0, verbose=1, random_state=0)
    Y = tsne.fit_transform(np.array(complete_samples_arr))
    print('embedding_: ******************************************')
    print(tsne.embedding_)
    print('kl_divergence_: ******************************************')
    print(tsne.kl_divergence_)
    print('n_iter_: ******************************************')
    print(tsne.n_iter_)
    end_time = time.time()
    print('TSNE time consumed:')
    print(end_time - start_time)
    # Assign the specific color to dimensionality reduced samples of each label
    df = pd.DataFrame(Y, columns=['x0', 'x1'])
    colors_dict = {'0.0': 'red', '1.0': 'blue', '2.0': 'green', '3.0': 'yellow', '4.0': 'black'}
    for index, row in df.iterrows():
        if index in range(0, len(train_labels)):
            df.at[index, 'label'] = train_labels[index]
        else:
            df.at[index, 'label'] = test_labels[index - len(train_labels)]
        current_color_key = str(df.at[index, 'label'])
        df.at[index, 'color'] = colors_dict[current_color_key]

    my_pic_title = dataset_name + ' by ' + rep_method
    draw_2d_with_colors(df, 'x0', 'x1', df['color'], my_pic_title)
    return


def draw_2d_with_colors(results_df, x_column, y_column, colors, pic_title):
    my_data_dict = dict()
    my_data_dict['x_label'], my_data_dict['y_label'] = x_column, y_column
    my_data_dict['x_values'], my_data_dict['y_values'] = results_df[x_column], results_df[y_column]
    draw_2D_figure(pic_title, my_data_dict, colors=colors)
    return


if __name__ == '__main__':
    dataset_name = 'Memo'
    # Option 1: summary the statistical characteristics of the dataset
    if False:
        trace_mct_list = list()
        app_trace_dict = load_transform_traces(dataset_name)
        for key in app_trace_dict.keys():
            my_mct = app_trace_dict.get(key)
            trace_mct_list.append(my_mct)
        draw_2d_mct_depth_nodes(dataset_name, trace_mct_list)

    # Option 2: visualize the representation generated by statistical representation method such as Histo, ECDF, etc.
    if True:
        mts_param_dict = {'etl_component': Local_Constants.MY_ETL_COMPONENTS[2]}
        my_generator = MTS_Representation_Generator({}, dataset_name, mts_param_dict)
        all_representation_dict = my_generator.get_all_representations_dict()

        for rep_key in all_representation_dict.keys():
            representation = all_representation_dict.get(rep_key)
            reduce_dim_and_show(dataset_name, 'Statistical RG_'+rep_key, representation)

    # Option 3: visualize the representation generated by semantic representation method
    if True:
        top_k_keywords_sizes = list(Local_Constants.TOP_K_KEYWORDS_LIST)
        fixed_matrix_sizes = [Local_Constants.FIXED_INPUT_MATRIX_SIZE for i in range(len(top_k_keywords_sizes))]
        vector_dimensionality_sizes = [int(fixed_matrix_sizes[i] / top_k_keywords_sizes[i]) for i in
                                       range(len(top_k_keywords_sizes))]

        mts_param_dict = {'etl_component': Local_Constants.MY_ETL_COMPONENTS[3],
                          'top_k_keywords_sizes': top_k_keywords_sizes,
                          'vector_dimensionality_sizes': vector_dimensionality_sizes}
        my_generator = Semantic_MTS_Representation_Generator({}, dataset_name, mts_param_dict)
        all_representation_dict = my_generator.get_all_representations_dict()
        for rep_key in all_representation_dict.keys():
            representation = all_representation_dict.get(rep_key)
            reduce_dim_and_show(dataset_name, 'Semantic RG_'+rep_key, representation)

    print()
