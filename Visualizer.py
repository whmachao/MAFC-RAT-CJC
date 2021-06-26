import numpy as np
import math
import os
import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import messagebox
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter.filedialog
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd
import time
import threading
import random
import utilities.Constants as Constants


class Visualizer(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master.title("Time Series Analysis @ WUHAN University")
        self.master.resizable(width=True, height=True)
        self.datasource = None

        menu = Menu(self.master)
        self.master.config(menu=menu)
        filemenu = Menu(menu)
        menu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Aggregated-Results", command=self.select_aggregated_results_file)
        filemenu.add_command(label="Test-Results", command=self.select_aggregated_results_file)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.master.quit)

        helpmenu = Menu(menu)
        menu.add_cascade(label="Help", menu=helpmenu)
        helpmenu.add_command(label="About...", command=self.About)

        chart_frame = Frame(self.master)
        chart_frame.pack(side=LEFT)
        self.my_figure = Figure(figsize=(8, 6), dpi=100)
        self.my_canvas = FigureCanvasTkAgg(self.my_figure, master=chart_frame)
        self.my_canvas.draw()
        self.my_canvas.get_tk_widget().pack(side=BOTTOM, expand=YES, fill=X)
        toolbar = NavigationToolbar2Tk(self.my_canvas, chart_frame)
        toolbar.pack(side=TOP, expand=YES, fill=X)
        toolbar.update()

        filter_frame1 = Frame(self.master)
        filter_frame1.pack()
        tk.Label(filter_frame1, height=2, text="Grouped By: ").pack(side=LEFT)
        gp_index1 = tk.StringVar()
        self.groupby_comb1 = ttk.Combobox(filter_frame1, width=26, textvariable=gp_index1, state='readonly')
        self.groupby_comb1.pack(side=LEFT)
        self.groupby_comb1.bind("<<ComboboxSelected>>", self.update_groupby_values)
        gp_values_index1 = tk.StringVar()
        self.groupby_values_comb1 = ttk.Combobox(filter_frame1, width=26, textvariable=gp_values_index1, state='readonly')
        self.groupby_values_comb1.pack(side=LEFT)

        filter_frame2 = Frame(self.master)
        filter_frame2.pack()
        tk.Label(filter_frame2, height=2, text="                   ").pack(side=LEFT)
        gp_index2 = tk.StringVar()
        self.groupby_comb2 = ttk.Combobox(filter_frame2, width=26, textvariable=gp_index2, state='readonly')
        self.groupby_comb2.pack(side=LEFT)
        self.groupby_comb2.bind("<<ComboboxSelected>>", self.update_groupby_values)
        gp_values_index2 = tk.StringVar()
        self.groupby_values_comb2 = ttk.Combobox(filter_frame2, width=26, textvariable=gp_values_index2, state='readonly')
        self.groupby_values_comb2.pack(side=LEFT)

        multi_charts_frame1 = Frame(self.master)
        multi_charts_frame1.pack()
        self.check_var1 = IntVar()
        self.multi_charts_button = tk.Checkbutton(multi_charts_frame1, text="Multi-Charts", variable=self.check_var1,
                                                  onvalue=1, offvalue=0).pack(side=LEFT)
        gp_index3 = tk.StringVar()
        self.groupby_comb3 = ttk.Combobox(multi_charts_frame1, width=26, textvariable=gp_index3, state='readonly')
        self.groupby_comb3.pack(side=LEFT)

        multi_charts_frame2 = Frame(self.master)
        multi_charts_frame2.pack()
        tk.Label(multi_charts_frame2, height=2, text="                       ").pack(side=LEFT)
        gp_index4 = tk.StringVar()
        self.groupby_comb4 = ttk.Combobox(multi_charts_frame2, width=26, textvariable=gp_index4, state='readonly')
        self.groupby_comb4.pack(side=LEFT)

        x_axis_frame = Frame(self.master)
        x_axis_frame.pack()
        tk.Label(x_axis_frame, height=2, text="X-axis: ").pack(side=LEFT)
        x_index = tk.StringVar()
        self.x_comb = ttk.Combobox(x_axis_frame, width=26, textvariable=x_index, state='readonly')
        self.x_comb.pack(side=LEFT)

        y_axis_frame = Frame(self.master)
        y_axis_frame.pack()
        tk.Label(y_axis_frame, height=2, text="Y-axis: ").pack(side=LEFT)
        y_index = tk.StringVar()
        self.y_comb = ttk.Combobox(y_axis_frame, width=26, textvariable=y_index, state='readonly')
        self.y_comb.pack(side=LEFT)

        z_axis_frame = Frame(self.master)
        z_axis_frame.pack()
        tk.Label(z_axis_frame, height=2, text="Z-axis: ").pack(side=LEFT)
        z_index = tk.StringVar()
        self.z_comb = ttk.Combobox(z_axis_frame, width=26, textvariable=z_index, state='readonly')
        self.z_comb.pack(side=LEFT)

        operator_frame = Frame(self.master)
        operator_frame.pack()
        redraw_2d_button = tk.Button(operator_frame, height=1, text="ReDraw 2D", command=self.draw_new_2d_picture)
        redraw_2d_button.pack(side=LEFT)
        redraw_3d_button = tk.Button(operator_frame, height=1, text="ReDraw 3D", command=self.draw_new_3d_picture)
        redraw_3d_button.pack(side=LEFT)

        info_frame = Frame(self.master)
        info_frame.pack()
        self.results_file_url = tk.Label(info_frame, text="No results file is selected!", width=60, height=2,
                                         wraplength=400, justify='left', anchor='w')
        self.results_file_url.pack(side=LEFT)
        self.pb_hD = ttk.Progressbar(info_frame, orient='horizontal', mode='indeterminate')
        self.pb_hD.pack(side=LEFT)
        self.pb_hD.start(interval=50)

    def update_groupby_values(self, *args):
        if len(self.groupby_comb1.get()) > 0:
            value_list = self.datasource[self.groupby_comb1.get()].tolist()
            distinct_value_list = list(set(value_list))
            distinct_value_list.sort()
            self.groupby_values_comb1['values'] = []
            self.groupby_values_comb1['values'] = distinct_value_list
            self.groupby_values_comb1.config(state='readonly')
            self.groupby_values_comb1.current(0)
        if len(self.groupby_comb2.get()) > 0:
            value_list = self.datasource[self.groupby_comb2.get()].tolist()
            distinct_value_list = list(set(value_list))
            distinct_value_list.sort()
            self.groupby_values_comb2['values'] = []
            self.groupby_values_comb2['values'] = distinct_value_list
            self.groupby_values_comb2.config(state='readonly')
            self.groupby_values_comb2.current(0)

    def select_aggregated_results_file(self):
        results_file = tkinter.filedialog.askopenfilename(initialdir=os.getcwd() + '/aggregated_results/')

        if len(results_file) != 0:
            try:
                self.datasource = pd.read_csv(results_file)
            except:
                raise ValueError('Error happens when loading aggregated results CSV file!')
            self.results_file_url.config(text=results_file)
            all_column_names = self.datasource.columns.values.tolist()

            self.groupby_comb1['values'] = []
            self.groupby_comb1['values'] = all_column_names
            self.groupby_comb1.config(state='readonly')
            self.groupby_comb1.current(0)

            self.groupby_comb2['values'] = []
            self.groupby_comb2['values'] = all_column_names
            self.groupby_comb2.config(state='readonly')
            self.groupby_comb2.current(0)

            self.groupby_comb3['values'] = []
            self.groupby_comb3['values'] = all_column_names
            self.groupby_comb3.config(state='readonly')
            self.groupby_comb3.current(0)

            self.groupby_comb4['values'] = []
            self.groupby_comb4['values'] = all_column_names
            self.groupby_comb4.config(state='readonly')
            self.groupby_comb4.current(0)

            self.update_groupby_values()

            self.x_comb['values'] = []
            self.x_comb['values'] = all_column_names
            self.x_comb.config(state='readonly')
            self.x_comb.current(0)

            self.y_comb['values'] = []
            self.y_comb['values'] = all_column_names
            self.y_comb.config(state='readonly')
            self.y_comb.current(0)

            self.z_comb['values'] = []
            self.z_comb['values'] = all_column_names
            self.z_comb.config(state='readonly')
            self.z_comb.current(0)

    def draw_2d(self, x_column, y_column, g_column_names, g_column_values, multi_charts_names):
        if self.datasource is None:
            messagebox.showerror("Error", "Datasource not initialized!")
            return

        if len(g_column_names) > 1 and len(set(g_column_names)) < len(g_column_names):
            messagebox.showerror("Error", "Duplicated Groupby columns!")
            return

        if self.check_var1.get() != 0 and len(multi_charts_names) > 1 and len(set(multi_charts_names)) < len(multi_charts_names):
            messagebox.showerror("Error", "Duplicated Multi-charts columns!")
            return

        grouped_df = self.datasource.groupby(g_column_names)
        results_df = None
        for name, group in grouped_df:
            if self.get_str_list_from_tuple(name) == g_column_values:
                results_df = group
        print(results_df)
        if results_df is None:
            messagebox.showerror("No qualified records", "No qualified records found in the trace!")
            return

        # to determine single chart or multiple charts drawing
        if self.check_var1.get() == 0:
            self.my_figure.clf()
            ax = self.my_figure.add_subplot(1, 1, 1)
            title = tuple([str(x) for x in g_column_values])
            param_dict = {'top': max(results_df[y_column]), 'bottom': min(results_df[y_column]), 'scaling': 0.2,
                          'row_num': 1, 'col_num': 1, 'title': title, 'statistic': 'max'}
            if max(results_df[y_column]) < 1.0 and min(results_df[y_column]) >= 0.0:
                param_dict = {'top': 1.0, 'bottom': 0.0, 'scaling': 0.0,
                              'row_num': 1, 'col_num': 1, 'title': title, 'statistic': 'max'}
            self.draw_2d_single_chart(ax, results_df, x_column, y_column, param_dict)
            self.my_figure.set_tight_layout(tight='True')
            self.my_figure.autofmt_xdate(bottom=0.2, rotation=45, ha='right', which=None)
            self.my_canvas.draw()
        else:
            self.my_figure.clf()
            self.draw_2d_multi_charts(results_df, x_column, y_column, multi_charts_names)
            self.my_figure.set_tight_layout(tight='True')
            self.my_figure.autofmt_xdate(bottom=0.2, rotation=45, ha='right', which=None)
            self.my_canvas.draw()

    # Rather than the scatter plot, we prefer the box and whisker plot for showing the impact of split_ratio, bins and
    # representation levels in binary distribution tree on the time series classification accuracy
    def draw_2d_single_chart(self, ax, results_df, x_column, y_column, param_dict):
        x, y = results_df[x_column], results_df[y_column]
        grouped_df = results_df.groupby([x_column])
        orig_boxplot_arr, xticks_arr = [], []
        for name, group in grouped_df:
            y_list = list(group[y_column])
            orig_boxplot_arr.append(y_list)
            xticks_arr.append(name)

        if not self.isNumberList(y):
            messagebox.showerror("Y-axis must be a number list!", "Y-axis must be a number list!")
            return

        boxplot_arr = []
        if pd.isna(orig_boxplot_arr).any():
            print(pd.isna(orig_boxplot_arr))
            messagebox.showerror("Error", "NaN value(s) found in boxplot_arr!")
            for item in orig_boxplot_arr:
                item_without_nan = [x for x in item if ~np.isnan(x)]
                boxplot_arr.append(item_without_nan)
        else:
            boxplot_arr = orig_boxplot_arr

        margin_width, label_tick_size, auto_scaling = 0.5, 18, 0.1
        row_number, col_number = param_dict['row_num'], param_dict['col_num']
        charts_number = row_number * col_number
        if 1 < charts_number < 10 or len(xticks_arr) > 10:
            label_tick_size = int(label_tick_size / 2)
        elif 9 < charts_number:
            label_tick_size = int(label_tick_size / pow(2, math.sqrt(charts_number))) + 6
        ax.boxplot(boxplot_arr, whis=[0, 100], widths=margin_width * 0.8, showmeans=True)
        ax.set_xlim(auto=True)
        ax.set_xticklabels(xticks_arr, fontdict={'fontsize': label_tick_size})
        ax.set_title(param_dict['title'], fontdict={'fontsize': label_tick_size})

        # max_y_arr = np.array([np.round(max(item_list), 2) for item_list in boxplot_arr if len(item_list)>0])
        # min_y_arr = np.array([np.round(min(item_list), 2) for item_list in boxplot_arr if len(item_list)>0])
        # bottom, top = min(min_y_arr) * (1 - auto_scaling), max(max_y_arr) * (1 + auto_scaling)
        scaling = param_dict['scaling']
        bottom, top = param_dict['bottom'] * (1 - scaling), param_dict['top'] * (1 + scaling)
        ax.set_ylim(bottom=bottom, top=top)
        if Constants.AXIS_SHOW_NAME_DICT.get(x_column) is not None:
            ax.set_xlabel(Constants.AXIS_SHOW_NAME_DICT.get(x_column), fontdict={'fontsize':label_tick_size})
        else:
            ax.set_xlabel(x_column, fontdict={'fontsize': label_tick_size})
        if Constants.AXIS_SHOW_NAME_DICT.get(y_column) is not None:
            ax.set_ylabel(Constants.AXIS_SHOW_NAME_DICT.get(y_column), fontdict={'fontsize':label_tick_size})
        else:
            ax.set_ylabel(y_column, fontdict={'fontsize': label_tick_size})
        yticks = []
        for tick in ax.get_yticks():
            yticks.append(round(tick, 2))
        ax.set_yticklabels(yticks, fontdict={'fontsize':label_tick_size})

        # plot upper labels such as the largest value to facilitate the comparison among different boxes
        upperLabels = []
        for item_list in boxplot_arr:
            if len(item_list) == 0:
                upperLabels.append('nan')
            elif param_dict['statistic'] == 'max':
                upperLabels.append(str(np.round(max(item_list), 3)))
            elif param_dict['statistic'] == 'min':
                upperLabels.append(str(np.round(min(item_list), 3)))
            elif param_dict['statistic'] == 'average':
                upperLabels.append(str(np.round(np.mean(item_list), 3)))
            elif param_dict['statistic'] == 'mean':
                upperLabels.append(str(np.round(np.median(item_list), 3)))
            else:
                upperLabels.append(str(np.round(max(item_list), 3)))

        for tick in range(len(xticks_arr)):
            if 1 < charts_number or len(xticks_arr) > 10:
                ax.text(tick + 1, top * 0.85, upperLabels[tick], horizontalalignment='center',
                        size='xx-small', weight='bold', color='royalblue')
            else:
                ax.text(tick + 1, top * 0.9, upperLabels[tick], horizontalalignment='center',
                        size='xx-large', weight='bold', color='royalblue')
        # y_max = round(y.max(), 3)
        # a.set_title("Best " + str(y_column) + " = " + str(y_max), fontdict={'fontsize':22})

        # self.my_figure.set_tight_layout(tight='True')
        # self.my_figure.autofmt_xdate(bottom=0.2, rotation=45, ha='right', which=None)
        # self.my_canvas.draw()

    def draw_2d_multi_charts(self, results_df, x_column, y_column, multi_charts_names):
        if len(multi_charts_names) != 2:
            messagebox.showerror('Error', 'Multi-charts MUST contain exactly 2 items!')
            return
        grouped_df = results_df.groupby(multi_charts_names)
        grouped_df_by_row = results_df.groupby(multi_charts_names[0])
        grouped_df_by_col = results_df.groupby(multi_charts_names[1])
        row_number, col_number = grouped_df_by_row.ngroups, grouped_df_by_col.ngroups


        axes = self.my_figure.subplots(row_number, col_number, sharex='col', sharey='none', squeeze=False)
        row_index, col_index = 0, 0
        for name, group in grouped_df:
            print(name)
            current_ax = axes[row_index, col_index]
            param_dict = {'top': max(results_df[y_column]), 'bottom': min(results_df[y_column]), 'scaling': 0.3,
                          'row_num': row_number, 'col_num': col_number, 'title': str(name), 'statistic': 'max'}
            self.draw_2d_single_chart(current_ax, group, x_column, y_column, param_dict)
            if col_index == col_number - 1:
                row_index += 1
                col_index = 0
            else:
                col_index += 1


    def draw_new_2d_picture(self):
        if "No results file is selected!" == self.results_file_url.cget('text'):
            messagebox.showerror('Error', 'You MUST select a aggregated results file first!')
            return
        else:
            x_column, y_column, z_column, g_column_names, g_column_values, multi_charts_names = self.get_all_fields_names_and_values()
            x_column_is_conflicted, y_column_is_conflicted, multi_charts_columns_are_conflicted = True, True, True
            try:
                g_column_names.index(x_column)
                messagebox.showerror('Error', 'Groupby columns must be different from X column!')
                return
            except:
                x_column_is_conflicted = False

            try:
                g_column_names.index(y_column)
                messagebox.showerror('Error', 'Groupby columns must be different from Y column!')
                return
            except:
                y_column_is_conflicted = False

            multi_charts_columns_are_conflicted = False
            if self.check_var1.get() != 0:
                for item in multi_charts_names:
                    try:
                        g_column_names.index(item)
                        messagebox.showerror('Error', 'Groupby columns must be different from multi-charts columns!')
                        return
                    except:
                        multi_charts_columns_are_conflicted = False

            if not x_column_is_conflicted and not y_column_is_conflicted and not multi_charts_columns_are_conflicted:
                self.draw_2d(x_column, y_column, g_column_names, g_column_values, multi_charts_names)

    def draw_new_3d_picture(self):
        messagebox.showerror('Error', '3D chart drawing is not implemented yet!')
        return

    def get_all_fields_names_and_values(self):
        g_column_names, g_column_values, multi_charts_names = [], [], []
        try:
            x_column = self.x_comb.get()
            y_column = self.y_comb.get()
            z_column = self.z_comb.get()
            g_column_names.extend([self.groupby_comb1.get(), self.groupby_comb2.get()])
            g_column_values.extend([self.groupby_values_comb1.get(), self.groupby_values_comb2.get()])
            multi_charts_names.extend([self.groupby_comb3.get(), self.groupby_comb4.get()])
        except:
            messagebox.showerror("Parameter Error", "Something wrong happens!")
            return

        return x_column, y_column, z_column, g_column_names, g_column_values, multi_charts_names

    def get_str_list_from_tuple(self, my_tuple):
        my_tuple_list = list(my_tuple)

        for i in range(len(my_tuple_list)):
            my_tuple_list[i] = str(my_tuple_list[i])
        return my_tuple_list

    def About(self):
        print("This is a simple example of a menu")

    # To check whether all elements in the given list are number-like such as Integer, Floating-point numbers
    def isNumberList(self, my_list):
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

    def removeNanValues(self, my_list):
        return [x for x in my_list if x != np.nan]


if __name__ == '__main__':
    root = tk.Tk()
    app = Visualizer(root)
    app.mainloop()
