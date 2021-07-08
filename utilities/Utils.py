import json
import os
import time
from scipy.stats import tmean, tvar, tmin, tmax, tstd, tsem
from scipy.stats import gmean, hmean, kurtosis, skew, variation
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from treelib import Tree
import re
import utilities.Constants as Constants
from pathlib import Path


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def get_maximum_from_lists(list_in_lists):
    max_value = np.max(list_in_lists)
    if isinstance(max_value, list):
        max_value = np.max(max_value)
    return max_value


def get_minimum_from_lists(list_in_lists):
    min_value = np.min(list_in_lists)
    if isinstance(min_value, list):
        min_value = np.min(min_value)
    return min_value


def get_dataframe_from_jsontrace(filePath):
    start_time = time.time()
    print(start_time)

    data = []
    with open(filePath, 'rb') as f:
        for line in f:
            # print(str(json_lines.index(line)) + '-th line of ' + str(len(json_lines)) + ' has been processed!')
            try:
                data.append(json.loads(line))
            except:
                continue
                # following codes will not be executed because of the existing of too many problematic lines in traces
                # some items in original traces are not properly contained by quotation marks
                # so far we identified "-inf", "inf", and "nan"
                if '-inf' in line:
                    # print('-inf')
                    line = line.replace('-inf', '"-inf"')
                    data.append(json.loads(line))
                elif 'inf' in line:
                    # print('inf')
                    line = line.replace('inf', '"inf"')
                    data.append(json.loads(line))
                elif 'nan' in line:
                    # print('nan')
                    line = line.replace('nan', '"nan"')
                    data.append(json.loads(line))


    print(str(np.round(time.time() - start_time,2)) + ' seconds consumed for loading trace: ' + filePath)

    # First check: whether traces of the original file equals to the sum of traces of TEN events
    original_trace_df = pd.DataFrame(data=data)

    if check_traces_euqal(original_trace_df):
        print('First check passed: traces of the original file equals to the sum of traces of TEN events!')
    else:
        raise ValueError('First check failed!')

    # Second check: whether traces of threadID 1 segment equals to the sum of traces of TEN events in the same segment
    thread_one_df = original_trace_df[original_trace_df['threadID'] == 1]

    if check_traces_euqal(thread_one_df):
        print('Second check passed: traces of thread one equals to the sum of traces of TEN events!')
    else:
        raise ValueError('First check failed!')

    # the following two lines make the thread_one_df drop lines in which the "event" value is 5
    # (i.e. "method" value is "null"). So far. it seems that the removal operation will not twist the call relations
    thread_one_df = thread_one_df[thread_one_df['event'] != 5]

    # replace items with different formats as those in the same format
    thread_one_df = thread_one_df.replace(np.nan, 'N/A', regex=True)
    thread_one_df = thread_one_df.replace('nan', 'N/A', regex=True)

    event_eight_df = original_trace_df[original_trace_df['event'] == 8]
    event_nine_df = original_trace_df[original_trace_df['event'] == 9]

    return thread_one_df, event_eight_df, event_nine_df


# construct the method call tree for each independent block
# 'Event' categorization: 0 - 9
# 0: Start; 1: Exceptional End; 2: Normal End; 3: Read Field; 4: Write Field; 5: Catch Exception; 6: Catch Notify
# 7: Catch Wait; 8: Method Name; 9: Field Name
def construct_method_call_tree(my_trace_name, my_trace_url):
    my_thread_one_df, my_event_eight_df, my_event_nine_df = get_dataframe_from_jsontrace(my_trace_url)
    my_tree = Tree()
    # create the root node: the entire trace
    my_root = my_tree.create_node(tag=my_trace_name, identifier=my_trace_name,
                                  data=[my_thread_one_df, my_event_eight_df, my_event_nine_df])
    next_level_method_call_df_list = get_next_level_method_call_block_list(my_thread_one_df)
    for method_call_df in next_level_method_call_df_list:
        # method_call_df = method_call_df.reset_index()
        # tag = method_call_df.loc[0]['method']
        # identifier = method_call_df.loc[0]['threadClock']
        check_method_call_block_df(method_call_df)
        construct_new_sub_tree(method_call_df, my_event_eight_df, my_event_nine_df, my_tree, my_trace_name)
    # my_tree.show()
    # print(my_trace_name + ' method call tree statistics:')
    # print('Depth: ' + str(my_tree.depth()))
    # print('No. of Nodes: ' + str(len(my_tree.all_nodes())))

    exceptional_lines = 0
    if my_tree.depth() > 0:
        level_one_nodes = my_tree.children(nid=my_trace_name)
        total_lines = len(my_thread_one_df)
        for node in level_one_nodes:
            current_node_thread_one_df = node.data[0]
            total_lines = total_lines - len(current_node_thread_one_df)
        if total_lines != 0:
            # raise ValueError('Expect total_lines to be 0 but got ' + str(total_lines))
            exceptional_lines = total_lines

    # only the tree structure is stored as txt, while the data of nodes are discarded
    # my_tree.save2file(Local_Constants.ROOT_DIR + '../AFD/images/' + my_trace_name.split('&')[-1].split('.')[0] + '.txt')
    return my_tree, exceptional_lines


def construct_new_sub_tree(method_call_df, method_name_df, field_name_df, current_tree, parent):
    row_count, row_number, first_row_index,  last_row_index= 0, len(method_call_df), None, None
    tag, identifier = None, None
    for index, row in method_call_df.iterrows():
        if row_count == 0:
            first_row_index = index
            filtered_df = method_name_df.loc[method_name_df.retVal == row['method']]
            if len(filtered_df) == 1:
                tag = filtered_df.iloc[0]['method']
            else:
                tag = row['method']
            identifier = row['threadClock']
            current_tree.create_node(tag, identifier, parent=parent, data=[method_call_df, method_name_df, field_name_df])
        if row_count == row_number-1:
            last_row_index = index
        row_count += 1
    new_method_call_df = method_call_df.drop([first_row_index, last_row_index])
    if len(new_method_call_df) == 0:
        return
    next_level_method_call_df_list = get_next_level_method_call_block_list(new_method_call_df)
    if len(next_level_method_call_df_list) == 0:
        return

    for smaller_method_call_df in next_level_method_call_df_list:
        construct_new_sub_tree(smaller_method_call_df, method_name_df, field_name_df, current_tree, identifier)


def get_nodes_at_specific_level(target_tree, target_level):
    if not isinstance(target_level, int) or target_level < 0 or target_level > target_tree.depth():
        raise ValueError('Expect tree_level as an integer in range from 0 to ' +
                         str(target_tree.depth()) + 'but got ' + str(target_level))
    return [node for node in target_tree.all_nodes_itr() if target_tree.level(node.identifier) == target_level]


# find out the next level method call blocks
def get_next_level_method_call_block_list(thread_one_df):
    method_call_block_df_list = []

    method_call_block_df = pd.DataFrame()
    top_method_flag = False
    top_method_name = ''
    for index, row in thread_one_df.iterrows():
        if row['event'] == 0 and not top_method_flag and top_method_name == '':
            top_method_name = row['method']
            top_method_flag = True
        if top_method_flag and top_method_name != '':
            method_call_block_df = method_call_block_df.append(row)
            if row['method'] == top_method_name and (row['event'] == 1 or row['event'] == 2):
                method_call_block_df_list.append(method_call_block_df)
                method_call_block_df = pd.DataFrame()
                top_method_flag, top_method_name = False, ''
    # print('Found ' + str(len(method_call_block_df_list)) + ' independent method call blocks!')

    # for my_df in method_call_block_df_list:
    #     my_df = my_df.reset_index(drop=True)
    #     print_full_dataframe(my_df)
    #     method, clock = my_df.loc[0]['method'], my_df.loc[0]['threadClock']
    #     print('Method: ' + str(method) + ', ' + 'Clock: ' + str(clock))

    return method_call_block_df_list


# pad the original method call tree to the specified level
def pad_method_call_tree(method_call_tree, target_tree_level):
    if not isinstance(method_call_tree, Tree):
        raise ValueError('Expect method_call_tree as a Tree, but got the type ' + type(method_call_tree))
    if not isinstance(target_tree_level, int) or target_tree_level < 1:
        raise ValueError('Expect target_tree_level as an integer, but got ' + str(target_tree_level))

    my_leaves = method_call_tree.leaves()
    for leaf_node in my_leaves:
        if method_call_tree.depth(leaf_node) >= target_tree_level:
            continue
        levels_to_be_paded = target_tree_level - method_call_tree.depth(leaf_node)
        new_tag = leaf_node.tag
        new_parent = leaf_node.identifier
        new_data = leaf_node.data
        for i in range(levels_to_be_paded):
            new_node = method_call_tree.create_node(new_tag, data=new_data, parent=new_parent)
            new_tag = new_node.tag
            new_parent = new_node.identifier
    return method_call_tree


def check_method_call_block_df(method_call_block_df):
    method_call_block_df = method_call_block_df.reset_index()

    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # print(method_call_block_df)

    num_rows = len(method_call_block_df)
    if method_call_block_df.loc[0]['method'] != method_call_block_df.loc[num_rows-1]['method']:
        raise ValueError('First line and last line method mismatched!')
    if method_call_block_df.loc[0]['event'] != 0:
        raise ValueError('First line event is not 0!')
    if method_call_block_df.loc[num_rows-1]['event'] != 1 and method_call_block_df.loc[num_rows-1]['event'] != 2:
        raise ValueError('Last line event is not 1 or 2!')


def get_call_graph_by_method(method, thread_one_df):
    # thread_one_df = thread_one_df.reset_index()
    # thread_one_df = thread_one_df.rename(columns={'index':'origin_line'})

    method_start = thread_one_df[(thread_one_df.method==method) & (thread_one_df.event == 0)].index.tolist()
    method_exceptional_end = thread_one_df[(thread_one_df.method==method) & (thread_one_df.event == 1)].index.tolist()
    method_normal_end = thread_one_df[(thread_one_df.method==method) & (thread_one_df.event == 2)].index.tolist()
    method_end = method_normal_end + method_exceptional_end
    method_start.sort()
    method_end.sort()

    method_call_details = []
    # handle the case that the method is in event_eight_df but not called in the entire trace
    if len(method_start) == 0 and len(method_end) == 0:
        method_call_details.append({'method':method, 'call_order':0, 'start_line':0, 'end_line':0,
                                    'start_clock':0, 'end_clock':0, 'call_segment':None})

    # handle the case that the method is in event_eight_df and called for at least once but start points do not equal to
    # end points, thus will be omitted.
    if len(method_start) != len(method_end):
        method_call_details.append({'method':method, 'call_order':np.inf, 'start_line':0, 'end_line':0,
                                    'start_clock':0, 'end_clock':0, 'call_segment':None})

    # handle the case that the method is in event_eight_df and called for at least once and start points equal to
    # end points, thus each call will be recorded in detail.
    if len(method_start) == len(method_end):
        for i in range(len(method_start)):
            start_line, end_line = method_start[i], method_end[i]
            start_clock, end_clock = thread_one_df.loc[start_line,'threadClock'], thread_one_df.loc[end_line,'threadClock']
            call_segment = thread_one_df.loc[start_line:end_line, :]
            method_call_details.append({'method': method, 'call_order': i + 1, 'start_line': start_line, 'end_line': end_line,
                                        'start_clock': start_clock, 'end_clock': end_clock, 'call_segment':call_segment})
    method_call_details_df = pd.DataFrame(data=method_call_details)
    return method_call_details_df


def draw_2D_figure(my_title, my_data_dict, colors=None, legend=None, y_log_scale=False, pic_size=(6, 6), pic_type=None):
    x_label, y_label = my_data_dict.get('x_label'), my_data_dict.get('y_label')
    x_values, y_values = my_data_dict.get('x_values'), my_data_dict.get('y_values')

    plt.rcParams['figure.figsize'] = pic_size
    fig, ax1 = plt.subplots()
    axis_font = {'fontname': 'Arial', 'size': '16'}
    ax1.set_xlabel(x_label, **axis_font)
    ax1.set_ylabel(y_label, **axis_font)
    if y_log_scale:
        ax1.set_yscale('log')
    points = None
    if pic_type is None:
        points = ax1.scatter(x_values, y_values, color=colors)
    elif pic_type == 'mutiple_lines':
        for order in range(int((len(x_values) + 1) / 2)):
            points = ax1.plot(x_values[order*2:order*2+2], y_values[order*2:order*2+2], c='r')
        my_x_ticks = get_k_split_ticks(x_values)
        my_y_ticks = get_k_split_ticks(y_values)
        plt.xticks(my_x_ticks, my_x_ticks)
        plt.yticks(my_y_ticks, my_y_ticks)
    elif pic_type == 'bars':
        plt.bar(x_values, y_values, width=0.3)
    else:
        raise ValueError('Unknown picture type: ' + pic_type)
    if legend is not None:
        my_legend = ax1.legend(handles=[points], labels=legend, loc='upper right', shadow=True, fontsize='x-large')
        # Put a nicer background color on the legend.
        my_legend.get_frame().set_facecolor('#eafff5')
    if '&' in my_title:
        output_directory = Constants.ROOT_DIR + 'AFD/images/' + my_title.split('&')[0] + '/'
        app_name = my_title.split('&')[1].split('.')[-1]
        my_title = app_name + '&' + my_title.split('&')[-1]
        save_url = output_directory + my_title + '.jpg'
    else:
        output_directory = Constants.ROOT_DIR + 'AFD/images/' + my_title + '/'
        save_url = output_directory + my_title + '.jpg'
    ax1.tick_params(axis='y')
    ax1.set_title(my_title, fontdict={'fontsize': 20})
    fig.autofmt_xdate(bottom=0.12, rotation=90, ha='right', which=None)
    plt.margins(0.01, 0.01)
    create_directory(output_directory)
    plt.savefig(save_url, bbox_inches='tight', dpi=1000)
    plt.clf()
    plt.close()


def get_callees_by_segment(temp_call_segment_df):
    normal_callees_list, exceptional_callees_list = [], []
    # for index, row in temp_call_segment_df.iterrows():
    #     print(index)
    method_start = temp_call_segment_df[temp_call_segment_df['event'] == 0].index.tolist()
    method_exceptional_end = temp_call_segment_df[temp_call_segment_df['event'] == 1].index.tolist()
    method_normal_end = temp_call_segment_df[temp_call_segment_df['event'] == 2].index.tolist()
    method_end = method_normal_end + method_exceptional_end
    method_start.sort()
    method_end.sort()
    # handle cases that the trace segment is incomplete
    if len(method_start) != len(method_end):
        for start_line in method_start[1:]:
            exceptional_callees_list.append(start_line)
        # print(temp_call_segment_df.columns.values)
        # print(temp_call_segment_df.values)
        # raise ValueError('Incomplete call traces!')
    # handle the case that no callee is contained in the trace segment
    if len(method_start) == 1 and len(method_end) == 1:
        return [], []
    # handle the case that at least one callee is contained in the trace segment
    if len(method_start) > 1 and len(method_end) > 1 and len(method_start) == len(method_end):
        for start_line in method_start[1:]:
            normal_callees_list.append(start_line)
            # my_callees_list.append(temp_call_segment_df.loc[index]['method'])
    return normal_callees_list, exceptional_callees_list


def check_traces_euqal(my_trace_df):
    check_flag = False

    event_zero_df = my_trace_df[my_trace_df['event'] == 0]
    event_one_df = my_trace_df[my_trace_df['event'] == 1]
    event_two_df = my_trace_df[my_trace_df['event'] == 2]
    event_three_df = my_trace_df[my_trace_df['event'] == 3]
    event_four_df = my_trace_df[my_trace_df['event'] == 4]
    event_five_df = my_trace_df[my_trace_df['event'] == 5]
    event_six_df = my_trace_df[my_trace_df['event'] == 6]
    event_seven_df = my_trace_df[my_trace_df['event'] == 7]
    event_eight_df = my_trace_df[my_trace_df['event'] == 8]
    event_nine_df = my_trace_df[my_trace_df['event'] == 9]

    if event_six_df.shape[0] > 0 or event_seven_df.shape[0] > 0:
        raise ValueError('Found that event 6 or event 7 traces without any processing logic!')

    all_events_traces = event_zero_df.shape[0] + event_one_df.shape[0] + event_two_df.shape[0] + event_three_df.shape[0] \
                        + event_four_df.shape[0] + event_five_df.shape[0] + event_six_df.shape[0] + event_seven_df.shape[0] \
                        + event_eight_df.shape[0] + event_nine_df.shape[0]

    if my_trace_df.shape[0] == all_events_traces:
        check_flag = True

    return check_flag


def check_continuity_among_idependent_calls(thread_one_df, hex_method_df):
    if thread_one_df.empty or hex_method_df.empty:
        return None
    thread_one_start_line, thread_one_end_line = thread_one_df.index[0], thread_one_df.index[-1]
    start_line = hex_method_df.loc[hex_method_df.index[0]]['start_line']
    end_line = hex_method_df.loc[hex_method_df.index[-1]]['end_line']
    if start_line != thread_one_start_line or end_line != thread_one_end_line + 1:
        return str(start_line) + ':' + str(end_line) + ' does not match ' + str(thread_one_start_line) + ':' + str(thread_one_end_line)

    # Detect whether there is any gap between two adjacent independent call segments
    last_end_line = None
    err_msg = ''
    for index, row in hex_method_df.iterrows():
        if last_end_line is None:
            last_end_line = row['end_line']
            continue
        if row['start_line'] != last_end_line:
            err_msg = err_msg + '\n' + 'current start_line ' + str(row['start_line']) + ' not equal to last end_line ' + str(last_end_line)
            last_end_line = row['end_line']
        else:
            last_end_line = row['end_line']

    if err_msg != '':
        return err_msg

    return None


def get_k_split_ticks(ori_values_list, k=10):
    if not isinstance(ori_values_list, list):
        raise ValueError('Expected a list object but got a ' + type(ori_values_list))
    if len(ori_values_list) <= k:
        return ori_values_list
    k_split_values_list = []
    delta = (max(ori_values_list) - min(ori_values_list)) / (k-1)
    for i in range(k):
        if i == k-1:
            k_split_values_list.append(int(max(ori_values_list)))
            break
        next_value = min(ori_values_list) + i*delta
        k_split_values_list.append(int(next_value))
    return k_split_values_list


def simple_histogram_transformation(lists_in_list, bin_num):
    if not isinstance(lists_in_list, list):
        raise ValueError('lists_in_list must be the type of list!')
    for my_list in lists_in_list:
        if len(my_list) == 0:
            raise ValueError('All sub-lists must be non-empty!')
    if not isinstance(bin_num, int):
        raise ValueError('bin_num must be the type of int!')
    if bin_num < 3:
        raise ValueError('bin_num must be greater than 2!')

    x_train_max, x_train_min = get_maximum_from_lists(lists_in_list), get_minimum_from_lists(lists_in_list)
    overall_bin_edges = [x_train_min]
    bin_width = (x_train_max - x_train_min) / bin_num
    for i in range(bin_num + 1):
        overall_bin_edges.append(overall_bin_edges[i] + bin_width)

    new_lists_in_list = []
    for my_list in lists_in_list:
        hist, bin_edges = np.histogram(my_list, bins=overall_bin_edges)
        new_lists_in_list.append(hist)
    print(new_lists_in_list)

    return new_lists_in_list


def simple_distribution_transform(lists_in_list):
    if not isinstance(lists_in_list, list):
        raise ValueError('lists_in_list must be the type of list!')
    for my_list in lists_in_list:
        if len(my_list) == 0:
            raise ValueError('All sub-lists must be non-empty!')

    new_lists_in_list = []
    for my_list in lists_in_list:
        my_tmean, my_tvar, my_tmin = tmean(my_list), tvar(my_list), tmin(my_list)
        my_tmax, my_tstd, my_tsem = tmax(my_list), tstd(my_list), tsem(my_list)
        my_gmean, my_kurtosis = gmean(my_list), kurtosis(my_list)
        my_skew, my_variation = skew(my_list), variation(my_list)
        new_lists_in_list.append([my_tmean, my_tvar, my_tmin, my_tmax, my_tstd, my_tsem,
                                  my_gmean, my_kurtosis, my_skew, my_variation])
    print(new_lists_in_list)

    return new_lists_in_list


# So far we do not know how to appropriately use empirical cumulative distribution function (a.k.a. ECDF)
def ecdf_based_transform(lists_in_list, data_points):
    if not isinstance(lists_in_list, list):
        raise ValueError('lists_in_list must be the type of list!')
    for my_list in lists_in_list:
        if len(my_list) == 0:
            raise ValueError('All sub-lists must be non-empty!')
    if not isinstance(data_points, int) and data_points > 3:
        raise ValueError('Constants.DATA_POINTS_NUMBER is wrongly set!')

    new_lists_in_list = []
    for my_list in lists_in_list:
        my_sampled_arr = get_sampled_arr_by_sliding_window(my_list, Constants.WINDOW_SIZE, Constants.WINDOW_STRIDE)
        my_ecdf_rep = ecdfRep(my_sampled_arr, data_points)
        new_lists_in_list.append(my_ecdf_rep)
    return new_lists_in_list


# Extract the array from the raw series by using a fixed size sliding window with a specific stride
def get_sampled_arr_by_sliding_window(raw_list, wnd_size, wnd_stride):
    if len(raw_list) < wnd_size:
        raise ValueError('raw_list size MUST be larger than wnd_size!')
    sampled_list = list()
    count = 0
    while count * wnd_stride + wnd_size <= len(raw_list):
        start_index = count * wnd_stride
        end_index = start_index + wnd_size
        my_slice = raw_list[start_index:end_index]
        sampled_list.append(my_slice)
        count += 1
    return np.array(sampled_list)


# From the paper "On preserving statistical characteristics of accelerometry data using their empirical cumulative distribution."
def ecdfRep(data, components):
    m = data.mean(0)
    data = np.sort(data, axis=0)
    data = data[np.int32(np.around(np.linspace(0, data.shape[0] - 1, num=components))), :]
    data = data.flatten(1)
    return np.hstack((data, m))


def new_process_single_trace(thread_one_df, method_name_df):
    thread_one_df = thread_one_df.reset_index()
    thread_one_df = thread_one_df.rename(columns={'index':'origin_line'})

    hex_method_name_list = list(method_name_df['retVal'])
    str_method_name_list = list(method_name_df['method'])
    if len(hex_method_name_list) != len(set(hex_method_name_list)):
        raise ValueError('There are duplicated hexdecimal method names in event eight traces!')
    if len(str_method_name_list) != len(set(str_method_name_list)):
        raise ValueError('There are duplicated string method names in event eight traces!')

    data = []
    start_line, end_line = None, None
    method_call_start_dict_list = []
    for index, row in thread_one_df.iterrows():
        if row['event'] == 0:
            method_call_start_dict = {'method':row['method'], 'line':index, 'clock':row['threadClock']}
            method_call_start_dict_list.append(method_call_start_dict)
        if row['event'] == 1 or row['event'] == 2:
            last_method_call_start_dict = method_call_start_dict_list[-1]
            if last_method_call_start_dict.get('method') == row['method']:
                del method_call_start_dict_list[-1]
                if len(method_call_start_dict_list) == 0:
                    start_line = last_method_call_start_dict.get('line')
                    start_clock = last_method_call_start_dict.get('clock')
                    end_line = index + 1
                    temp_call_segment = thread_one_df.iloc[start_line:end_line, :]
                    temp_dict = {'method':row['method'], 'start_line': start_line, 'end_line': end_line,
                                 'call_segment': temp_call_segment, 'call_segment_size': temp_call_segment.shape[0],
                                 'start_clock':start_clock, 'end_clock':row['threadClock']}
                    data.append(temp_dict)


    hex_method_df = pd.DataFrame(data=data)

    check_result = check_continuity_among_idependent_calls(thread_one_df, hex_method_df)
    if check_result is not None:
        print('Warning when checking continuity: ' + check_result)

    return hex_method_df

# 获取某个dataset下的X_Train,Y_Train,X_Test,Y_Test，堆属性进行了拼接
# 输入为指定dataset的文件夹路径
# 输出为四个list类型数据
def getData(dirPath):
    # 读取数据
    dataNameList = os.listdir(dirPath)
    # print(dataNameList)
    trainData = []
    testData = []
    trainLabel = []
    testLabel = []
    dataNum = 0
    for dataName in dataNameList:
        dataNameSplitList = str(dataName).split('_')
        # print(dataNameSplitList)
        if dataNameSplitList[1] == "TEST.txt":
            flag = 0
            attributeDataList = []
            path = os.path.join(dirPath, dataName)
            f = open(path, 'rb')
            for line in f:
                dataNum += 1
                line = str(line)
                dataStr = line[2:]
                dataStr = dataStr.replace(r"\n", '')
                dataStr = dataStr.replace(r"\r", '')
                dataStr = dataStr.replace(" ", '')
                dataStr = dataStr.replace("'", '')
                dataStrList = dataStr.split(',')
                # print(dataStrList)
                dataListFloat = []
                for item in dataStrList[1:]:
                    dataListFloat.append(float(item))
                flag += 1
                if flag == 4:
                    attributeDataList += dataListFloat
                    label = float(dataStrList[0])
                    # print('label:', label)
                    testLabel.append(label)
                    testData.append(attributeDataList)
                    flag = 0
                    attributeDataList = []
                else:
                    attributeDataList += dataListFloat
        elif dataNameSplitList[1] == "TRAIN.txt":
            flag = 0
            attributeDataList = []
            path = os.path.join(dirPath, dataName)
            f = open(path, 'rb')
            for line in f:
                dataNum += 1
                line = str(line)
                dataStr = line[2:]
                dataStr = dataStr.replace(r"\n", '')
                dataStr = dataStr.replace(r"\r", '')
                dataStr = dataStr.replace(" ", '')
                dataStr = dataStr.replace("'", '')
                dataStrList = dataStr.split(',')
                # print(dataStrList)
                dataListFloat = []
                for item in dataStrList[1:]:
                    dataListFloat.append(float(item))
                flag += 1
                if flag == 4:
                    attributeDataList += dataListFloat
                    label = float(dataStrList[0])
                    # print('label:', label)
                    trainLabel.append(label)
                    # print(trainLabel)
                    trainData.append(attributeDataList)
                    flag = 0
                    attributeDataList = []
                else:
                    attributeDataList += dataListFloat
    # print('数据总条数:', dataNum / 4)
    return trainLabel, trainData, testLabel, testData

# 按属性读取数据，没有拼接
def getArrtributeData(dirPath):
    # 读取数据
    dataNameList = os.listdir(dirPath)
    # print(dataNameList)
    trainData = []
    testData = []
    trainLabel = []
    testLabel = []
    dataNum = 0
    for dataName in dataNameList:
        dataNameSplitList = str(dataName).split('_')
        # print(dataNameSplitList)
        if dataNameSplitList[1] == "TEST.txt":
            flag = 0
            path = os.path.join(dirPath, dataName)
            oneSeries={
                'attribute0':[],
                'attribute1':[],
                'attribute2':[],
                'attribute3':[]
            }
            f = open(path, 'rb')
            for line in f:
                dataNum += 1
                line = str(line)
                dataStr = line[2:]
                dataStr = dataStr.replace(r"\n", '')
                dataStr = dataStr.replace(r"\r", '')
                dataStr = dataStr.replace(" ", '')
                dataStr = dataStr.replace("'", '')
                dataStrList = dataStr.split(',')
                # print(dataStrList)
                dataListFloat = []
                for item in dataStrList[1:]:
                    dataListFloat.append(float(item))
                flag += 1
                if flag == 4:
                    theKey = 'attribute' + str(flag-1)
                    oneSeries[theKey] = dataListFloat
                    label = float(dataStrList[0])
                    testLabel.append(label)
                    testData.append(oneSeries)
                    flag = 0
                    oneSeries = {
                        'attribute0': [],
                        'attribute1': [],
                        'attribute2': [],
                        'attribute3': []

                    }
                else:
                    theKey = 'attribute' + str(flag - 1)
                    oneSeries[theKey] = dataListFloat
        elif dataNameSplitList[1] == "TRAIN.txt":
            flag = 0
            oneSeries = {
                'attribute0': [],
                'attribute1': [],
                'attribute2': [],
                'attribute3': []

            }
            path = os.path.join(dirPath, dataName)
            f = open(path, 'rb')
            for line in f:
                dataNum += 1
                line = str(line)
                dataStr = line[2:]
                dataStr = dataStr.replace(r"\n", '')
                dataStr = dataStr.replace(r"\r", '')
                dataStr = dataStr.replace(" ", '')
                dataStr = dataStr.replace("'", '')
                dataStrList = dataStr.split(',')
                # print(dataStrList)
                dataListFloat = []
                for item in dataStrList[1:]:
                    dataListFloat.append(float(item))
                flag += 1
                if flag == 4:
                    theKey='attribute'+str(flag-1)
                    oneSeries[theKey]=dataListFloat

                    label = float(dataStrList[0])
                    # print(label)
                    trainLabel.append(label)
                    trainData.append(oneSeries)
                    flag = 0
                    oneSeries = {
                        'attribute0': [],
                        'attribute1': [],
                        'attribute2': [],
                        'attribute3': []

                    }
                else:
                    theKey='attribute'+str(flag-1)
                    oneSeries[theKey]=dataListFloat

    return trainLabel, trainData, testLabel, testData


# output indices of train set and test set according to the split ratio and original labels' list
def get_train_test_indices(my_split_ratio, my_labels_list):
    if my_split_ratio <= 0.0 or my_split_ratio >=1.0:
        raise ValueError('Expect my_split_ratio in range (0.0, 1.0) but got ' + my_split_ratio)
    my_unique_labels_list = list(set(my_labels_list))
    my_train_indices, my_test_indices = [], []
    if Constants.SPLIT_BY_APP:
        split_index = int(len(my_unique_labels_list) * my_split_ratio)
        for my_unique_label in my_unique_labels_list:
            my_indices = [index for index, my_label in enumerate(my_labels_list) if my_label == my_unique_label]
            if my_unique_labels_list.index(my_unique_label) < split_index:
                my_train_indices = my_train_indices + my_indices
            else:
                my_test_indices = my_test_indices + my_indices
    else:
        for my_unique_label in my_unique_labels_list:
            my_indices = [index for index, my_label in enumerate(my_labels_list) if my_label == my_unique_label]
            split_index = int(len(my_indices) * my_split_ratio)
            if split_index < 1:
                my_train_indices.append(my_indices[0])
                my_test_indices = my_test_indices + my_indices[1:]
            elif split_index >= len(my_indices):
                my_train_indices = my_train_indices + my_indices[1:]
                my_test_indices.append(my_indices[0])
            else:
                my_train_indices = my_train_indices + my_indices[:split_index]
                my_test_indices = my_test_indices + my_indices[split_index:]

    return my_train_indices, my_test_indices


def create_classifier(classifier_name, output_directory, param_dict):
    if classifier_name == 'FCN':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, param_dict['input_shape'], param_dict['nb_classes'], Constants.VERBOSE)
    if classifier_name == 'MLP':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, param_dict['input_shape'], param_dict['nb_classes'], Constants.VERBOSE)
    if classifier_name == 'ResNet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, param_dict['input_shape'], param_dict['nb_classes'], Constants.VERBOSE)
    if classifier_name == 'LSTM':
        from classifiers import lstm
        return lstm.Classifier_LSTM(output_directory, param_dict['input_shape'], param_dict['nb_classes'], Constants.VERBOSE)
    if classifier_name == 'KNN':
        from classifiers import knn
        return knn.Classifier_KNN(output_directory, param_dict['k_value'], param_dict['pred_strategy'], param_dict['distance_metric'])
    raise ValueError(classifier_name + ' is not the supported CLASSIFIER!')

# def create_classifier(classifier_name, input_shape, nb_classes, output_directory):
#     if classifier_name == 'FCN':
#         from classifiers import fcn
#         return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, Constants.VERBOSE)
#     if classifier_name == 'MLP':
#         from classifiers import mlp
#         return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, Constants.VERBOSE)
#     if classifier_name == 'ResNet':
#         from classifiers import resnet
#         return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, Constants.VERBOSE)
#     if classifier_name == 'LSTM':
#         from classifiers import lstm
#         return lstm.Classifier_LSTM(output_directory, input_shape, nb_classes, Constants.VERBOSE)
#     if classifier_name == 'KNN':
#         from classifiers import knn
#         return knn.Classifier_KNN(output_directory, None, None, Constants.VERBOSE)
#     raise ValueError(classifier_name + ' is not the supported CLASSIFIER!')


def create_representation_generator(representation_method_name, datasets_dict, dataset_name):
    if representation_method_name == Constants.MY_REPRESENTORS[1]:
        from representors import level_histo_representor
        level_histo_param_dict = {'tree_level_list': Constants.TREE_LEVEL_LIST,
                                  'fixed_length_list': Constants.FIXED_LENGTH_LIST,
                                  'gram_num_list': Constants.GRAM_NUMBER_LIST,
                                  'etl_component': Constants.MY_ETL_COMPONENTS[1]}
        return level_histo_representor.Level_Histo_Representor(datasets_dict, dataset_name, level_histo_param_dict)
    if representation_method_name == Constants.MY_REPRESENTORS[2]:
        from representors import mts_representor
        mts_param_dict = {'etl_component': Constants.MY_ETL_COMPONENTS[2],
                          'data_points_list': Constants.DATA_POINTS_LIST}
        return mts_representor.MTS_Representor(datasets_dict, dataset_name, mts_param_dict)
    if representation_method_name == Constants.MY_REPRESENTORS[3]:
        from representors import semantic_mts_representor
        top_k_keywords_sizes = list(Constants.TOP_K_KEYWORDS_LIST)
        fixed_matrix_sizes = [Constants.FIXED_INPUT_MATRIX_SIZE for i in range(len(top_k_keywords_sizes))]
        vector_dimensionality_sizes = [int(fixed_matrix_sizes[i]/top_k_keywords_sizes[i]) for i in range(len(top_k_keywords_sizes))]

        sem_mts_param_dict = {'etl_component': Constants.MY_ETL_COMPONENTS[3],
                              'top_k_keywords_sizes': top_k_keywords_sizes,
                              'vector_dimensionality_sizes': vector_dimensionality_sizes}
        return semantic_mts_representor.Semantic_MTS_Representor(datasets_dict, dataset_name, sem_mts_param_dict)
    if representation_method_name == Constants.MY_REPRESENTORS[4]:
        from representors import feature_finder_representor
        top_k_keywords_sizes = list(Constants.TOP_K_KEYWORDS_LIST)
        feature_finder_param_dict = {'etl_component': Constants.MY_ETL_COMPONENTS[3],
                                     'top_k_keywords_sizes': top_k_keywords_sizes}
        return feature_finder_representor.Feature_Finder_Representor(datasets_dict, dataset_name, feature_finder_param_dict)
    raise ValueError(representation_method_name + ' is not a supported REPRESENTATION_GENERATOR!')


def create_etl_component(etl_component_name, method_call_tree, param_dict):
    if etl_component_name == Constants.MY_ETL_COMPONENTS[0]:
        from etl import plain_histo_ETL
        return plain_histo_ETL.Plain_Histo_Extractor(method_call_tree, 1)
    if etl_component_name == Constants.MY_ETL_COMPONENTS[1]:
        from etl import level_histo_ETL
        tree_level, n_gram = param_dict['tree_level'], param_dict['n_gram']
        return level_histo_ETL.Level_Histo_Extractor(method_call_tree, tree_level, n_gram)
    if etl_component_name == Constants.MY_ETL_COMPONENTS[2]:
        from etl import method_call_tree_mts_ETL
        return method_call_tree_mts_ETL.Method_Call_Tree_MTS_Extractor(method_call_tree, None, None)
    if etl_component_name == Constants.MY_ETL_COMPONENTS[3]:
        from etl import method_call_tree_sementics_mts_ETL
        k_keywords = param_dict['k_keywords']
        return method_call_tree_sementics_mts_ETL.Method_Call_Tree_Semantic_MTS_Extractor(method_call_tree, k_keywords, None)
    raise ValueError(etl_component_name + ' is not a supported ETL_COMPONENT!')


def print_full_dataframe(my_df):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    print(my_df)

    return


# split the whole dataset into the train/test sets
def split_train_test(true_labels_list, split_labels_list, num_of_attributes, hist_arr_in_lists_by_attribute):
    train_samples_list, train_labels_list, test_samples_list, test_labels_list = None, None, None, None

    # if Local_Constants.LEAVE_ONE_APP_OUT_AS_TEST:
    #     return train_samples_list, train_labels_list, test_samples_list, test_labels_list

    train_test_split_ratio = Constants.TRAIN_TEST_SPLIT_RATIO
    # Obtain indices for train and test sets
    train_indices, test_indices = get_train_test_indices(train_test_split_ratio, split_labels_list)
    train_samples_list, train_labels_list, test_samples_list, test_labels_list = [], [], [], []
    for train_index in train_indices:
        train_sample = []
        for attribute_index in range(num_of_attributes):
            train_sample = train_sample + list(hist_arr_in_lists_by_attribute[attribute_index][train_index])
        train_samples_list.append(train_sample)
        train_labels_list.append(true_labels_list[train_index])
    for test_index in test_indices:
        test_sample = []
        for attribute_index in range(num_of_attributes):
            test_sample = test_sample + list(hist_arr_in_lists_by_attribute[attribute_index][test_index])
        test_samples_list.append(test_sample)
        test_labels_list.append(true_labels_list[test_index])
    return train_samples_list, train_labels_list, test_samples_list, test_labels_list


def get_normalized_list(my_list):
    maximum, minimum = max(my_list), min(my_list)
    normalized_list = list()
    for item in my_list:
        if maximum == minimum:
            normalized_value = 1 / len(my_list)
        else:
            normalized_value = (item - minimum) / (maximum - minimum)
        normalized_list.append(normalized_value)
    return normalized_list


# load the English words dictionary
def load_words_dict(words_txt_url):
    with open(words_txt_url) as word_file:
        valid_words = set(word_file.read().split())

    return valid_words


# Extract all human readable words (sub-strings) from the given string
def extract_words(words_set, ori_string, max_window_size=8, min_length=3, delimiter='.'):
    if not isinstance(ori_string, str):
        raise ValueError('Expect ori_string as a non-empty string, but got ' + str(ori_string))
    if not isinstance(max_window_size, int) or max_window_size < 4:
        raise ValueError('Expect max_window_size as a integer greater than 3, but got ' + str(max_window_size))
    if not isinstance(min_length, int) or min_length < 3:
        raise ValueError('Expect min_length as a integer greater than 2, but got ' + str(min_length))
    if max_window_size < min_length:
        raise ValueError('Expect max_window_size is larger than min_length!')
    my_words_list = list()
    candidate_list = ori_string.split(delimiter)
    # Remove those candidates with the length less than 3
    for candidate in candidate_list:
        if len(candidate) < min_length:
            candidate_list.remove(candidate)

    for candidate in candidate_list:
        camel_case_words = camel_case_split(candidate)
        if len(camel_case_words) > 0:
            for word in camel_case_words:
                if word.lower() in words_set:
                    my_words_list.append(word)
            continue
        # for window_size in range(min_length, max_window_size):
        #     for index in range(len(candidate) - window_size):
        #         sub_string = candidate[index:index+window_size]
        #         if candidate.lower() in words_set:
        #             my_words_list.append(sub_string)
    return my_words_list


# split the camel case string into words
def camel_case_split(str):
    return re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', str)


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['val_acc'].idxmax()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 7), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch',
                                          'time_consumption_in_seconds'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['acc']
    df_best_model['best_model_val_acc'] = row_best_model['val_acc']
    df_best_model['time_consumption_in_seconds'] = duration
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_metrics.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics


def delete_logs(output_directory):
    if os.path.exists(output_directory):
        names = os.listdir(output_directory)
        for name in names:
            if not (name.endswith('.csv') or name == 'test' or name == 'train'):
                os.remove(output_directory + name)
    else:
        return None


def get_optimal_batch_size(train_set_size, default_batch_size, percentage):
    if not isinstance(train_set_size, int) or not isinstance(default_batch_size, int):
        raise ValueError('train_set_size and default_batch_size must be Integers!')
    if not isinstance(percentage, float):
        raise ValueError('percentage must be Float!')
    if train_set_size < 1 or default_batch_size < 1:
        raise ValueError('train_set_size and default_batch_size must be greater than 0!')
    if percentage not in np.round(list(np.arange(0.01, 1.01, 0.01)),2):
        raise ValueError('percentage must be in the range [0.01, 1.00] with stride 0.01!')

    candidate_batch_size = train_set_size * percentage
    if candidate_batch_size < 2:
        return 1

    return min(2**int(np.log2(candidate_batch_size)), default_batch_size)


# To help all the other python code easily obtain the project root directory.
def get_project_dir():
    project_root = Path(__file__).parent.parent
    return str(project_root)


if __name__ == '__main__':
    get_project_dir()

    print(get_sampled_arr_by_sliding_window([1,2,3,4,5,6,7], 3, 2))
    create_representation_generator(Constants.MY_REPRESENTATION_GENERATORS[3], None, None)

    my_words = load_words_dict('words.txt')
    # test_string = 'void com.abhi.newmemo.adapter.MemoAdapter.onItemDismiss'
    # test_string = 'com.abhi.newmemo.adapter.MemoAdapter'
    test_string = 'boolean com.abhi.newmemo.model.Memo.isArchived'
    words_list = extract_words(my_words, test_string)

    tree = Tree()
    tree.create_node("Harry", "harry", data='my_harry')  # root node
    tree.create_node("Jane", "jane", data='my_jane', parent="harry")
    tree.create_node("Bill", "bill", data='my_bill', parent="harry")
    tree.create_node("Diane", "diane", data='my_diane', parent="jane")
    tree.create_node("Mary", "mary", data='my_mary', parent="diane")
    tree.create_node("Mark", "mark", data='my_mark', parent="jane")
    tree.show()
    print('Depth: ' + str(tree.depth()))
    print('No. of Nodes: ' + str(len(tree.all_nodes())))
    my_leaves = tree.leaves()

    padded_tree = pad_method_call_tree(tree, Constants.PADDING_METHOD_CALL_TREE)
    padded_tree.show()
    print('Depth: ' + str(padded_tree.depth()))
    print('No. of Nodes: ' + str(len(padded_tree.all_nodes())))

    RAW_TRACE_DIR = Constants.ROOT_DIR + '/archives/APP_TRACE_Archive_2019'
    category_names_list = os.listdir(RAW_TRACE_DIR)
    title = 'Method Call Tree'
    data_dict = {'x_label': 'Tree Depth', 'x_values': [], 'y_label': 'No. of Nodes', 'y_values': []}
    for category_name in category_names_list:
        if category_name != 'Memo':
            continue
        category_dir = RAW_TRACE_DIR + '/' + category_name + '/'
        app_names_list = os.listdir(category_dir)
        for app_name in app_names_list:
            if app_name != 'com.abhi.newmemo':
                continue
            traceDir = category_dir + app_name
            trace_name_list = os.listdir(traceDir)
            for trace_name in trace_name_list:
                if trace_name != 'delete_memo-1.trace':
                    continue
                trace_path = RAW_TRACE_DIR + '/' + category_name + '/' + app_name + '/' + trace_name
                call_tree, wrong_lines = construct_method_call_tree(category_name+'&' + app_name + '&' + trace_name, trace_path)
                data_dict['x_values'].append(call_tree.depth())
                data_dict['y_values'].append(len(call_tree.all_nodes()))
    draw_2D_figure(title, data_dict)

