import os
import json
import pandas as pd
import bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Spectral4


def plotting_bokeh(path_to_data, file_names, colours, *TOOLTIPS, **plotconfigs):
    file_data = [pd.read_csv(x, usecols=["Step", "Value"]) for x in path_to_data]
    TOOLTIPS = TOOLTIPS[0]
    plotconfigs = plotconfigs["plotconfigs"]

    p = figure(plot_width=plotconfigs["plot_width"],
               plot_height=plotconfigs["plot_height"],
               tooltips=TOOLTIPS)
    p.title.text = plotconfigs["plot_title"]

    for data, name, color in zip(file_data, file_names, colours):
        p.line(data['Step'], data['Value'], line_width=2, color=color, alpha=0.8, legend=name)

    p.legend.location = plotconfigs["plot_legend_location"]
    p.legend.click_policy = plotconfigs["plot_legend_click_policy"]

    output_file(plotconfigs["output_file_name"],
                title=plotconfigs["output_file_title"])

    show(p)


if __name__ == "__main__":

    # showing loss plot
    # getting paths
    filenames_loss = [x for x in os.listdir(os.path.join(
        os.getcwd(), 'plots')) if x[-3:] == 'csv' and ('loss' in x or 'MSE' in x)]
    filepath_loss = [os.path.join(os.getcwd(), 'plots', x) for x in filenames_loss]

    # getting plotting config
    plotting_config = 'config/plot_loss.json'
    print('plotting from config file {}'.format(os.path.abspath(plotting_config)))
    with open(plotting_config) as json_file:
        plot_config = json.load(json_file)
    print(plot_config)
    # setting tooltips for bokeh plot
    TOOLTIPS = [("index", "$index")]
    TOOLTIPS.append((plot_config["TOOLTIPS1"], plot_config["TOOLTIPS2"]))
    print(TOOLTIPS)
    plotting_bokeh(filepath_loss,
                   filenames_loss,
                   Spectral4,
                   TOOLTIPS,
                   plotconfigs=plot_config)

    # showing acc plot
    # getting paths
    filenames_acc = [x for x in os.listdir(os.path.join(
        os.getcwd(), 'plots')) if x[-3:] == 'csv' and ('acc' in x or 'f1' in x)]
    filepath_acc = [os.path.join(os.getcwd(), 'plots', x) for x in filenames_acc]

    # getting plotting config
    plotting_config = 'config/plot_acc.json'
    print('plotting from config file {}'.format(os.path.abspath(plotting_config)))
    with open(plotting_config) as json_file:
        plot_config = json.load(json_file)
    print(plot_config)
    # setting tooltips for bokeh plot
    TOOLTIPS = [("index", "$index")]
    TOOLTIPS.append((plot_config["TOOLTIPS1"], plot_config["TOOLTIPS2"]))
    print(TOOLTIPS)
    plotting_bokeh(filepath_acc,
                   filenames_acc,
                   Spectral4,
                   TOOLTIPS,
                   plotconfigs=plot_config)
