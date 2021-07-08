import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import math
import os
import utilities.Constants as Constants
from matplotlib.figure import Figure
from statistics import mean
from utilities.Utils import create_directory
from wordcloud import WordCloud
from representors.semantic_mts_representor import Semantic_MTS_Representor
from sklearn.metrics import silhouette_samples, silhouette_score
import scipy.stats as stats

def compute_overall_results(round_decimals = 3):
    # compute the statistics including the average accuracy, standard deviation in Table 3 of the manuscript
    random = np.array([0.333, 0.25, 0.167, 0.077])
    print('Statistics of random')
    print(np.mean(random).round(round_decimals), np.std(random, ddof=1).round(round_decimals))

    fcn_scr = np.array([0.556, 0.542, 0.609, 0.322])
    print('Statistics of fcn_scr')
    print(np.mean(fcn_scr).round(round_decimals), np.std(fcn_scr, ddof=1).round(round_decimals))

    lstm_scr = np.array([0.833, 0.562, 0.55, 0.309])
    print('Statistics of lstm_scr')
    print(np.mean(lstm_scr).round(round_decimals), np.std(lstm_scr, ddof=1).round(round_decimals))

    mlp_scr = np.array([0.444, 0.5, 0.413, 0.311])
    print('Statistics of mlp_scr')
    print(np.mean(mlp_scr).round(round_decimals), np.std(mlp_scr, ddof=1).round(round_decimals))

    resnet_scr = np.array([0.611, 0.562, 0.543, 0.3])
    print('Statistics of resnet-scr')
    print(np.mean(resnet_scr).round(round_decimals), np.std(resnet_scr, ddof=1).round(round_decimals))

    fcn_sr = np.array([0.667, 0.854, 0.674, 0.467])
    print('Statistics of fcn_sr')
    print(np.mean(fcn_sr).round(round_decimals), np.std(fcn_sr, ddof=1).round(round_decimals))

    lstm_sr = np.array([0.778, 0.604, 0.646, 0.33])
    print('Statistics of lstm_sr')
    print(np.mean(lstm_sr).round(round_decimals), np.std(lstm_sr, ddof=1).round(round_decimals))

    mlp_sr = np.array([0.889, 0.625, 0.63, 0.478])
    print('Statistics of mlp_sr')
    print(np.mean(mlp_sr).round(round_decimals), np.std(mlp_sr, ddof=1).round(round_decimals))

    resnet_sr = np.array([0.778, 0.771, 0.609, 0.511])
    print('Statistics of resnet_sr')
    print(np.mean(resnet_sr).round(round_decimals), np.std(resnet_sr, ddof=1).round(round_decimals))

    rat_optimals = np.array([0.889, 0.854, 0.674, 0.511])
    print('Statistics of rat_optimals')
    print(np.mean(rat_optimals).round(round_decimals), np.std(rat_optimals, ddof=1).round(round_decimals))

    return


def draw_multi_bars(save_dir, title, data_list, xticklabels_list, xlabel, ylabel, legend=None, pic_size=(6, 6)):
    for bar_list in data_list:
        if len(bar_list) != len(xticklabels_list):
            raise ValueError('All elements of data_list must have the same length as xlabels_list!')
    plt.rcParams['figure.figsize'] = pic_size
    fig, ax1 = plt.subplots()
    axis_font = {'fontname': 'Arial', 'size': '16'}
    ax1.set_xlabel(xlabel, **axis_font)
    ax1.set_ylabel(ylabel, **axis_font)


    index = np.arange(len(xticklabels_list))
    bar_width = 1 / (len(data_list) + 1)
    for item_index in range(len(data_list)):
        bar_offset = item_index * bar_width
        # a "CN" color spec, i.e. 'C' followed by a number, which is an index into the default property cycle
        my_color = 'C' + str(item_index)
        plt.bar(index + bar_offset, data_list[item_index], color=my_color, width=bar_width)

    plt.legend(legend)
    ax1.set_title(title)
    ax1.set_xticklabels([''] + xticklabels_list)
    fig.autofmt_xdate(bottom=0.12, rotation=30, ha='right', which=None)
    plt.margins(0.01, 0.01)
    plt.savefig(save_dir + title + '.pdf', bbox_inches='tight', dpi=1000)

    return


# Everything looks fine except the position of the group of boxes are not aligned to the x-ticks
def draw_2d_box_plots_in_single_chart(group_df, atrrs_list, param_dict, pic_size=(8, 6)):
    plt.rcParams['figure.figsize'] = pic_size
    fig, ax1 = plt.subplots()
    axis_font = {'fontname': 'Arial', 'size': '16'}
    x_column, y_column = atrrs_list[0], atrrs_list[1]
    ax1.set_xlabel(x_column, **axis_font)
    ax1.set_ylabel(y_column, **axis_font)
    legend = list()

    xticks_arr = []
    yticks = group_df[y_column]
    if not isNumberList(yticks):
        raise ValueError("Y-axis must be a number list!")
    subgroup_df = group_df.groupby(x_column)
    if len(atrrs_list) == 2:  # If only 2 strings in atrrs_list, there will be the single box at each x-tick;
        boxplot_arr = []
        for name, group in subgroup_df:
            y_list = list(group[y_column])
            boxplot_arr.append(y_list)
            xticks_arr.append(name)

        if pd.isna(boxplot_arr).any():
            raise ValueError("NaN value(s) found in boxplot_arr!")
        plt.boxplot(boxplot_arr, whis=[0, 100], widths=0.6, showmeans=True)
    elif len(atrrs_list) == 3:  # If 3 strings in atrrs_list, there will be the double boxes at each x-tick;
        boxplot_arr_list = []
        z_attr = atrrs_list[2]
        z_attr_group_df = group_df.groupby(z_attr)
        for z_attr_name, z_attr_group in z_attr_group_df:
            boxplot_arr = []
            x_attr_group_df = z_attr_group.groupby(x_column)
            for x_attr_name, x_attr_group in x_attr_group_df:
                y_list = list(x_attr_group[y_column])
                boxplot_arr.append(y_list)
                if len(xticks_arr) < len(x_attr_group_df):
                    xticks_arr.append(x_attr_name)
            boxplot_arr_list.append(boxplot_arr)
            legend.append(z_attr_name)

        my_index = 0
        boxes_in_group = len(boxplot_arr_list)
        width = 1 / boxes_in_group
        delta = 2 * width / boxes_in_group
        for boxplot_arr in boxplot_arr_list:
            poistions = np.arange(my_index*width+delta, len(boxplot_arr)+my_index*width+delta, 1)
            bp = plt.boxplot(boxplot_arr, positions=poistions, widths=width*0.6, whis=[0, 100])
            set_box_color(bp, 'C'+str(my_index+1))
            my_index += 1
        ax1.legend(legend, loc='upper left')
    else:
        raise ValueError('Expect atrrs_list having 2 or 3 items but got ' + str(len(atrrs_list)))

    ax1.set_title(param_dict['title'])
    ax1.set_xticklabels(xticks_arr)
    ax1.set_ylim(bottom=param_dict['bottom'], top=param_dict['top'])
    fig.autofmt_xdate(bottom=0.12, rotation=30, ha='right', which=None)
    plt.margins(0.01, 0.01)
    my_results_dir_url = os.getcwd() + '/images/' + param_dict['title'] + '.pdf'
    plt.savefig(my_results_dir_url, bbox_inches='tight', dpi=1000)


# Set the color for each component of the boxplot bp
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


# Draw a group of bars at each x-tick
def draw_2d_multi_bars_at_each_xtick(group_df, atrrs_list, param_dict, pic_size=(6, 6)):
    if len(atrrs_list) != 3:
        raise ValueError('Expect atrrs_list having 3 items, but got ' + str(len(atrrs_list)))
    plt.rcParams['figure.figsize'] = pic_size
    fig, ax1 = plt.subplots()
    axis_font = {'fontname': 'Arial', 'size': '20'}
    x_column, y_column = atrrs_list[0], atrrs_list[1]
    if Constants.AXIS_SHOW_NAME_DICT.get(x_column) is not None:
        ax1.set_xlabel(Constants.AXIS_SHOW_NAME_DICT.get(x_column), **axis_font)
    else:
        ax1.set_xlabel(x_column, **axis_font)
    if Constants.AXIS_SHOW_NAME_DICT.get(y_column) is not None:
        ax1.set_ylabel(Constants.AXIS_SHOW_NAME_DICT.get(y_column), **axis_font)
    else:
        ax1.set_ylabel(y_column, **axis_font)
    legend = list()

    xticks_arr = []
    yticks = group_df[y_column]
    if not isNumberList(yticks):
        raise ValueError("Y-axis must be a number list!")
    bars_arr_list = []
    z_attr = atrrs_list[2]
    z_attr_group_df = group_df.groupby(z_attr)
    for z_attr_name, z_attr_group in z_attr_group_df:
        boxplot_arr = []
        x_attr_group_df = z_attr_group.groupby(x_column)
        for x_attr_name, x_attr_group in x_attr_group_df:
            # y_list_mean = mean(list(x_attr_group[y_column]))
            # boxplot_arr.append(y_list_mean)
            y_list_max = max(list(x_attr_group[y_column]))
            boxplot_arr.append(y_list_max)
            if len(xticks_arr) < len(x_attr_group_df):
                xticks_arr.append(x_attr_name)
        bars_arr_list.append(boxplot_arr)
        if Constants.LEGEND_SHOW_NAME_DICT.get(z_attr_name) is None:
            legend.append(z_attr_name)
        else:
            legend.append(Constants.LEGEND_SHOW_NAME_DICT.get(z_attr_name))
        # legend.append(z_attr_name)

    # draw bars in a group by group way
    index = np.arange(len(xticks_arr))
    bar_width = 1 / (len(bars_arr_list) + 1)
    for item_index in range(len(bars_arr_list)):
        bar_offset = item_index * bar_width
        # a "CN" color spec, i.e. 'C' followed by a number, which is an index into the default property cycle
        my_color = 'C' + str(item_index)
        plt.bar(index + bar_offset, bars_arr_list[item_index], color=my_color, width=bar_width)

    # draw text information (e.g. y-axis value of each bar)
    for xtick in xticks_arr:
        print('Add code to draw text over bars!')

    plt.legend(legend, loc='upper left')
    # ax1.set_title(param_dict['title'])
    ax1.set_xticklabels([''] + xticks_arr)
    ax1.set_ylim(bottom=param_dict['bottom'], top=param_dict['top'])
    fig.autofmt_xdate(bottom=0.12, rotation=30, ha='right', which=None)
    plt.margins(0.01, 0.01)
    my_results_dir_url = param_dict['results_dir'] + param_dict['title'] + '.pdf'
    plt.savefig(my_results_dir_url, bbox_inches='tight', dpi=1000)


def draw_wordcloud(group_df, kw_column, weight_column, param_dict):
    my_keywords = ' '.join(list(group_df[kw_column]))
    my_wordcloud = WordCloud(background_color='white').generate_from_text(my_keywords)

    plt.imshow(my_wordcloud)
    plt.axis('off')
    # plt.show()

    my_results_file_url = param_dict['results_dir'] + param_dict['title'] + '.png'
    my_wordcloud.to_file(my_results_file_url)


def draw_silhouette_score(category_name):
    # Step 1: Generate the semantics based representation of the category 'category_name'
    top_k_keywords_sizes = Constants.TOP_K_KEYWORDS_LIST
    fixed_matrix_sizes = [Constants.FIXED_INPUT_MATRIX_SIZE for i in range(len(top_k_keywords_sizes))]
    vector_dimensionality_sizes = [int(fixed_matrix_sizes[i] / top_k_keywords_sizes[i]) for i in
                                   range(len(top_k_keywords_sizes))]

    mts_param_dict = {'etl_component': Constants.MY_ETL_COMPONENTS[3],
                      'top_k_keywords_sizes': top_k_keywords_sizes,
                      'vector_dimensionality_sizes': vector_dimensionality_sizes}
    my_generator = Semantic_MTS_Representor({}, category_name, mts_param_dict)
    all_representation_dict = my_generator.get_all_representations_dict()

    # Step 2: Compute and record the silhouette scores of all representations
    xtick_labels, train_silhouette_scores, test_silhouette_scores = list(), list(), list()
    for representation_key in all_representation_dict.keys():
        x_train = all_representation_dict[representation_key][0]
        y_train = all_representation_dict[representation_key][1]
        x_test = all_representation_dict[representation_key][2]
        y_test = all_representation_dict[representation_key][3]

        xtick_labels.append(representation_key)
        train_silhouette_scores.append(silhouette_score(x_train, y_train))
        test_silhouette_scores.append(silhouette_score(x_test, y_test))
        # print("For representation key:", representation_key,
        #       "For n_clusters =", len(set(y_train)),
        #       "The average silhouette_score is :", silhouette_avg)
        # print('******************************************************')

    # Step 3: Draw grouped bars for the train set and the test set
    results_dir = os.getcwd() + '/images/silhouette/'
    create_directory(results_dir)
    title = category_name
    data_list = [train_silhouette_scores, test_silhouette_scores]
    xlabel, ylabel = 'Representation Key', 'Silhouette Score'
    legend = ['Train Set', 'Test Set']
    draw_multi_bars(results_dir, title, data_list, xtick_labels, xlabel, ylabel, legend)

    print()


# Draw model loss on train/test set according to the history.csv
def draw_model_loss(model_loss_dir, results_dir):
    history_csv_files = os.listdir(model_loss_dir)
    for file in history_csv_files:
        if file.split('.')[-1] != 'csv':
            continue
        history_df = pd.read_csv(model_loss_dir + file)
        epoch_as_xticks = list(range(1, len(history_df)+1))
        train_loss_list, test_loss_list = list(history_df['loss']), list(history_df['val_loss'])

        save_fig_url = results_dir + file.replace('.csv', '.pdf')
        data_list = [train_loss_list, test_loss_list]
        xlabel, ylabel = 'Epoch', 'Loss'
        legend = ['Train Set', 'Test Set']
        draw_multi_lines(save_fig_url, '', data_list, epoch_as_xticks, xlabel, ylabel, legend)

    return


def draw_multi_lines(save_fig_url, title, data_list, xtick_labels, xlabel, ylabel, legend=None, pic_size=(6, 6)):
    for plot_list in data_list:
        if len(plot_list) != len(xtick_labels):
            raise ValueError('All elements of data_list must have the same length as xlabels!')
    plt.rcParams['figure.figsize'] = pic_size
    fig, ax1 = plt.subplots()
    axis_font = {'fontname': 'Arial', 'size': '16'}
    ax1.set_xlabel(xlabel, **axis_font)
    ax1.set_ylabel(ylabel, **axis_font)

    for item_index in range(len(data_list)):
        # a "CN" color spec, i.e. 'C' followed by a number, which is an index into the default property cycle
        my_color = 'C' + str(item_index)
        plt.plot(data_list[item_index], c=my_color)

    plt.legend(legend, loc='upper left')
    if title != '':
        ax1.set_title(title)
    # ax1.set_xticklabels(xtick_labels)
    fig.autofmt_xdate(bottom=0.12, rotation=30, ha='right', which=None)
    plt.margins(0.01, 0.01)
    plt.savefig(save_fig_url, bbox_inches='tight', dpi=1000)

    return

# To check whether all elements in the given list are number-like such as Integer, Floating-point numbers
def isNumberList(my_list):
    isNumberList = True
    my_set = set(my_list)
    for unqiue_item in iter(my_set):
        if isinstance(unqiue_item, int):
            continue
        elif isinstance(unqiue_item, float):
            continue
        else:
            isNumberList = False
            break
    return isNumberList


def tuple_to_string(my_tuple):
    my_string = ''
    for item in my_tuple:
        my_string += str(item)
    return my_string


def draw_ecdf_example(d_points):
    n = np.arange(-50, 50)
    mean = 0
    normal = stats.norm.pdf(n, mean, 10)
    plt.plot(n, normal)
    plt.xlabel('Distribution', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title("Normal Distribution")
    plt.show()

    accumulated_probability_values = list()
    original_probalility_values = list(normal)
    for probability in original_probalility_values:
        if len(accumulated_probability_values) == 0:
            accumulated_probability_values.append(probability)
        else:
            accumulated_probability_values.append(accumulated_probability_values[-1] + probability)
    plt.plot(n, accumulated_probability_values)
    plt.xlabel('Distribution', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title("Cummulated Distribution")
    plt.show()
    return

if __name__ == '__main__':
    my_results_url = os.getcwd() + '/aggregated_results/all_dl_results_2020_6_4_21_20-SCR-SR.csv'

    if False:
        draw_ecdf_example([])

    if False:
        model_loss_dir = os.getcwd() + '/model_loss/Memo/'
        results_dir = os.getcwd() + '/images/model-loss/'
        create_directory(results_dir)
        draw_model_loss(model_loss_dir, results_dir)

    if True:
        compute_overall_results()

    if False:
        draw_silhouette_score('Silhouette')

    if False:
        my_results_df = pd.read_csv('keywords.csv')
        attrs_for_grouping = ['trace', 'dimensionality']
        grouped_df = my_results_df.groupby(attrs_for_grouping)
        my_atrrs = ['dimensionality', 'tfidf', 'keyword']
        for name, group in grouped_df:
            print(name)
            results_dir = os.getcwd() + '/images/keywords/'
            create_directory(results_dir)
            param_dict = {'title': tuple_to_string(name), 'top': 1.0, 'bottom': 0.0, 'results_dir': results_dir}
            draw_wordcloud(group, 'keyword', 'tfidf', param_dict)

    if False:
        my_results_df = pd.read_csv(my_results_url)
        row_column_attrs = ['dataset', 'classifier']
        grouped_df = my_results_df.groupby(row_column_attrs)
        my_atrrs = ['dimensionality', 'best_model_val_acc', 'representation_generator']
        for name, group in grouped_df:
            print(name)
            results_dir = os.getcwd() + '/images/scr-sr-comparison/'
            create_directory(results_dir)
            param_dict = {'title': ''.join(name), 'top': 1.0, 'bottom': 0.0, 'results_dir': results_dir}
            draw_2d_multi_bars_at_each_xtick(group, my_atrrs, param_dict)

    if False:
        my_results_df = pd.read_csv(my_results_url)
        row_column_attrs = ['dataset', 'classifier']
        grouped_df = my_results_df.groupby(row_column_attrs)
        my_atrrs = ['dimensionality', 'best_model_val_acc', 'representation_generator']
        # my_atrrs = ['dimensionality', 'best_model_val_acc']
        for name, group in grouped_df:
            print(name)
            param_dict = {'title': str(name), 'top': 1.0, 'bottom': 0.0}
            draw_2d_box_plots_in_single_chart(group, my_atrrs, param_dict)

    if False:
        raw_end_1nn = [109, 5, 247, 272, 1003, 237]
        raw_edge_1nn = [66, 2.6, 129, 144, 536, 132]
        bdt_end_1nn = [69, 7.5, 87, 719, 668, 230]
        bdt_edge_1nn = [27, 2.1, 34, 85, 127, 42]
        onenn_data_list = []
        onenn_data_list.append(raw_end_1nn)
        onenn_data_list.append(raw_edge_1nn)
        onenn_data_list.append(bdt_end_1nn)
        onenn_data_list.append(bdt_edge_1nn)
        my_xlabels = ['Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'Phoneme', 'ScreenType']
        my_legend = ['RAW_END_1NN', 'RAW_EDGE_1NN', 'BDT_END_1NN', 'BDT_EDGE_1NN']
        results_dir = os.getcwd() + '/images/test/'
        create_directory(results_dir)
        draw_multi_bars(results_dir, 'Classifier-1NN', onenn_data_list, my_xlabels, 'Datasets', 'Time', my_legend)


