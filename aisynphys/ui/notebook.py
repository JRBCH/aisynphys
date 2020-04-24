"""Commonly used routines when working with jupyter / matplotlib
"""
import numpy as np
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt


def heatmap(data, row_labels, col_labels, ax=None, ax_labels=None, bg_color=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    
    Modified from https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    ax_labels
        (x, y) axis labels
    bg_color
        Background color shown behind transparent pixels
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    if bg_color is not None:
        bg = np.empty(data.shape[:2] + (3,))
        bg[:] = matplotlib.colors.to_rgb(bg_color)        
        ax.imshow(bg)
        
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)

    if ax_labels is not None:
        ax.set_ylabel(ax_labels[1], size=16)
        ax.set_xlabel(ax_labels[0], size=16)
        ax.xaxis.set_label_position('top')
    
    return im, cbar


def annotate_heatmap(im, labels, data=None, textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Modified from https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    labels
        Array of strings to display in each cell
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    pixels, _, _, _ = im.make_image(renderer=None, unsampled=True)

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            px_color = pixels[i,j]
            kw['color'] =  textcolors[int(np.mean(px_color[:3]) < 128)]
            text = im.axes.text(j, i, labels[i, j], **kw)
            texts.append(text)

    return texts


def show_connectivity_matrix(ax, results, pre_cell_classes, post_cell_classes, class_labels, cmap, norm):
    """Display a connectivity matrix.

    This function uses matplotlib to display a heatmap representation of the output generated by
    aisynphys.connectivity.measure_connectivity(). Each element in the matrix is colored by connection 
    probability, and the connection probability confidence interval is used to set the transparency 
    such that the elements with the most data (and smallest confidence intervals) will appear
    in more bold colors. 

    Parameters
    ----------
    ax : matplotlib.axes
        The matplotlib axes object on which to draw the connectivity matrix
    results : dict
        Output of aisynphys.connectivity.measure_connectivity. This structure maps
        (pre_class, post_class) onto the results of the connectivity analysis.
    pre_cell_classes : list
        List of presynaptic cell classes in the order they should be displayed
    post_cell_classes : list
        List of postsynaptic cell classes in the order they should be displayed
    class_labels : dict
        Maps {cell_class: label} to give the strings to display for each cell class.
    cmap : matplotlib colormap instance
        The colormap used to generate colors for each matrix element
    norm : matplotlib normalize instance
        Normalize instance used to normalize connection probability values before color mapping

    """
    # convert dictionary of results to a 2d array of connection probabilities
    shape = (len(pre_cell_classes), len(post_cell_classes))
    cprob = np.zeros(shape)
    cprob_alpha = np.zeros(shape)
    cprob_str = np.zeros(shape, dtype=object)

    for i,pre_class in enumerate(pre_cell_classes):
        for j,post_class in enumerate(post_cell_classes):
            result = results[pre_class, post_class]
            cp, cp_lower_ci, cp_upper_ci = result['connection_probability']
            cprob[i,j] = cp
            cprob_str[i,j] = "" if result['n_probed'] == 0 else "%d/%d" % (result['n_connected'], result['n_probed'])
            cprob_alpha[i,j] = 1.0 - 2.0 * (cp_upper_ci - cp_lower_ci)

    # map connection probability to RGB colors
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cprob_rgba = mapper.to_rgba(np.clip(cprob, 0.01, 1.0))

    # apply alpha based on confidence intervals
    cprob_rgba[:, :, 3] = np.clip(cprob_alpha, 0, 1)

    # generate lists of labels to display along the pre- and postsynaptic axes
    pre_class_labels = [class_labels[cls] for cls in pre_cell_classes]
    post_class_labels = [class_labels[cls] for cls in post_cell_classes]

    # draw the heatmap with axis labels and colorbar
    im, cbar = heatmap(cprob_rgba, pre_class_labels, post_class_labels, ax=ax, 
        ax_labels=('postsynaptic', 'presynaptic'),
        bg_color=(0.7, 0.7, 0.7),
        cmap=cmap, norm=norm, 
        cbarlabel="Connection probability", 
        cbar_kw={'shrink':0.5})

    # draw text over each matrix element
    labels = annotate_heatmap(im, cprob_str, data=cprob)
    
    return im, cbar, labels


def cell_class_matrix(pre_classes, post_classes, metric, class_labels, ax, db, pair_query_args=None):
    pair_query_args = pair_query_args or {}

    metrics = {
        #                               name                         unit   scale  db columns                                    colormap      log     clim           text format
        'psp_amplitude':               ('PSP Amplitude',             'mV',  1e3,   [db.Synapse.psp_amplitude],                   'bwr',        False,  (-1, 1),       "%0.2f mV"),
        'psp_rise_time':               ('PSP Rise Time',             'ms',  1e3,   [db.Synapse.psp_rise_time],                   'viridis_r',  False,  (0, 6),        "%0.2f ms"),
        'psp_decay_tau':               ('PSP Decay Tau',             'ms',  1e3,   [db.Synapse.psp_decay_tau],                   'viridis_r',  False,  (0, 20),       "%0.2f ms"),
        'psc_amplitude':               ('PSC Amplitude',             'mV',  1e3,   [db.Synapse.psc_amplitude],                   'bwr',        False,  (-1, 1),       "%0.2f mV"),
        'psc_rise_time':               ('PSC Rise Time',             'ms',  1e3,   [db.Synapse.psc_rise_time],                   'viridis_r',  False,  (0, 6),        "%0.2f ms"),
        'psc_decay_tau':               ('PSC Decay Tau',             'ms',  1e3,   [db.Synapse.psc_decay_tau],                   'viridis_r',  False,  (0, 20),       "%0.2f ms"),
        'latency':                     ('Latency',                   'ms',  1e3,   [db.Synapse.latency],                         'viridis_r',  False,  (0, 6),        "%0.2f ms"),
        'stp_initial_50hz':            ('Paired pulse STP',          '',    1,     [db.Dynamics.stp_initial_50hz],               'bwr',        False,  (-0.5, 0.5),   "%0.2f"),
        'stp_induction_50hz':          ('Train induced STP',         '',    1,     [db.Dynamics.stp_induction_50hz],             'bwr',        False,  (-0.5, 0.5),   "%0.2f"),
        'stp_recovery_250ms':          ('STP Recovery',              '',    1,     [db.Dynamics.stp_recovery_250ms],             'bwr',        False,  (-0.5, 0.5),   "%0.2f"),
        'pulse_amp_90th_percentile':   ('PSP Amplitude 90th %%ile',  'mV',  1e3,   [db.Dynamics.pulse_amp_90th_percentile],      'bwr',        False,  (-1, 1),       "%0.2f mV"),
    }
    metric_name, units, scale, columns, cmap, cmap_log, clim, cell_fmt = metrics[metric]

    pairs = db.matrix_pair_query(
        pre_classes=pre_classes,
        post_classes=post_classes,
        pair_query_args=pair_query_args,
        columns=columns,
    )

    pairs_has_metric = pairs[~pairs[metric].isnull()]
    metric_data = pairs_has_metric.groupby(['pre_class', 'post_class']).aggregate(lambda x: np.mean(x))

    cmap = matplotlib.cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1], clip=False)

    shape = (len(pre_classes), len(post_classes))
    data = np.zeros(shape)
    data_alpha = np.zeros(shape)
    data_str = np.zeros(shape, dtype=object)

    for i, pre_class in enumerate(pre_classes):
        for j, post_class in enumerate(post_classes):
            try:
                value = getattr(metric_data.loc[pre_class].loc[post_class], metric)
            except KeyError:
                value = np.nan
            data[i, j] = value * scale
            data_str[i, j] = cell_fmt % (value * scale) if np.isfinite(value) else ""
            data_alpha[i, j] = 1 if np.isfinite(value) else 0 
            
    pre_labels = [class_labels[cls] for cls in pre_classes]
    post_labels = [class_labels[cls] for cls in post_classes]
    mapper = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    data_rgb = mapper.to_rgba(data)
    data_rgb[:,:,3] = np.clip(data_alpha, 0, 1)

    im, cbar = heatmap(data_rgb, pre_labels, post_labels,
                    ax=ax,
                    ax_labels=('postsynaptic', 'presynaptic'),
                    bg_color=(0.8, 0.8, 0.8),
                    cmap=cmap,
                    norm=norm,
                    cbarlabel=metric_name,
                    cbar_kw={'shrink':0.5})

    text = annotate_heatmap(im, data_str, data=data)


def show_distance_profiles(ax, results, colors, class_labels):
    """ Display connection probability vs distance plots
    Parameters
    -----------
    ax : matplotlib.axes
        The matplotlib axes object on which to make the plots
    results : dict
        Output of aisynphys.connectivity.measure_distance. This structure maps
        (pre_class, post_class) onto the results of the connectivity as a function of distance.
    colors: dict
        color to draw each (pre_class, post_class) connectivity profile. Keys same as results.
        To color based on overall connection probability use color_by_conn_prob.
    class_labels : dict
        Maps {cell_class: label} to give the strings to display for each cell class.
    """

    for i, (pair_class, result) in enumerate(results.items()):
        pre_class, post_class = pair_class
        plot = ax[i]
        xvals = result['bin_edges']
        xvals = (xvals[:-1] + xvals[1:])*0.5e6
        cp = result['conn_prob']
        lower = result['lower_ci']
        upper = result['upper_ci']

        color = colors[pair_class]
        color2 = list(color)
        color2[-1] = 0.2
        mid_curve = plot.plot(xvals, cp, color=color, linewidth=2.5)
        lower_curve = plot.fill_between(xvals, lower, cp, color=color2)
        upper_curve = plot.fill_between(xvals, upper, cp, color=color2)
        
        plot.set_title('%s -> %s' % (class_labels[pre_class], class_labels[post_class]))
        if i == len(ax)-1:
            plot.set_xlabel('Distance (um)')
            plot.set_ylabel('Connection Probability')
        
    return ax


def color_by_conn_prob(pair_group_keys, connectivity, norm, cmap):
    """ Return connection probability mapped color from show_connectivity_matrix
    """
    colors = {}
    for key in pair_group_keys:
        cp = connectivity[key]['connection_probability'][0]
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        color = mapper.to_rgba(np.clip(cp, 0.01, 1.0))
        colors[key] = color

    return colors


def data_matrix(data_df, cell_classes, metric=None, scale=1, unit=None, cmap=None, norm=None, alpha=2):
    """ Return data and labels to make a matrix using heatmap and annotate_heatmap. Similar to 
    show_connectivity_matrix but for arbitrary data metrics.

    Parameters:
    -----------
    data_df : pandas dataframe 
        pairs with various metrics as column names along with the pre-class and post-class.
    cell_classes : list 
        cell classes included in the matrix, this assumes a square matrix.
    metric : str
        data metric to be displayed in matrix
    scale : float
        scale of the data
    unit : str
        unit for labels
    cmap : matplotlib colormap instance
        used to colorize the matrix
    norm : matplotlib normalize instance
        used to normalize the data before colorizing
    alpha : int
        used to desaturate low confidence data
    """

    shape = (len(cell_classes), len(cell_classes))
    data = np.zeros(shape)
    data_alpha = np.zeros(shape)
    data_str = np.zeros(shape, dtype=object)
    
    mean = data_df.groupby(['pre_class', 'post_class']).aggregate(lambda x: np.mean(x))
    error = data_df.groupby(['pre_class', 'post_class']).aggregate(lambda x: np.std(x))
    count = data_df.groupby(['pre_class', 'post_class']).count()
    
    for i, pre_class in enumerate(cell_classes):
        for j, post_class in enumerate(cell_classes):
            try:
                value = mean.loc[pre_class].loc[post_class][metric]
                std = error.loc[pre_class].loc[post_class][metric]
                n = count.loc[pre_class].loc[post_class][metric]
                if n == 1:
                    value = np.nan
                #data_df.loc[pre_class].loc[post_class][metric]
            except KeyError:
                value = np.nan
            data[i, j] = value*scale
            data_str[i, j] = "%0.2f %s" % (value*scale, unit) if np.isfinite(value) else ""
            data_alpha[i, j] = 1-alpha*((std*scale)/np.sqrt(n)) if np.isfinite(value) else 0 

    mapper = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    data_rgb = mapper.to_rgba(data)
    max = mean[metric].max()*scale
    data_rgb[:,:,3] = np.clip(data_alpha, 0, max)
    return data_rgb, data_str