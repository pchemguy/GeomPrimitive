# https://matplotlib.org/stable/users/explain/colors/colormaps.html
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html#sphx-glr-gallery-lines-bars-and-markers-multicolored-line-py
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html#sphx-glr-gallery-lines-bars-and-markers-fill-between-demo-py

from colorspacious import cspace_converter

import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl


cmaps = {}

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(category, cmap_list):
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)
    axs[0].set_title(f'{category} colormaps', fontsize=14)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=mpl.colormaps[name])
        ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()

    # Save colormap list for later.
    cmaps[category] = cmap_list


if __name__ == "__main__":
    plot_color_gradients('Perceptually Uniform Sequential',
                         ['viridis', 'plasma', 'inferno', 'magma', 'cividis'])
    plot_color_gradients('Sequential',
                         ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                          'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                          'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])
    plot_color_gradients('Sequential (2)',
                         ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                          'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                          'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'])
    plot_color_gradients('Diverging',
                         ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                          'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
                          'berlin', 'managua', 'vanimo'])
    plot_color_gradients('Cyclic', ['twilight', 'twilight_shifted', 'hsv'])
    plot_color_gradients('Qualitative',
                         ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                          'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
                          'tab20c'])
    plot_color_gradients('Miscellaneous',
                         ['flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                          'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
                          'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
                          'turbo', 'nipy_spectral', 'gist_ncar'])
    plt.show()

    ###############################################################################

    mpl.rcParams.update({'font.size': 12})
    
    # Number of colormap per subplot for particular cmap categories
    _DSUBS = {'Perceptually Uniform Sequential': 5, 'Sequential': 6,
              'Sequential (2)': 6, 'Diverging': 6, 'Cyclic': 3,
              'Qualitative': 4, 'Miscellaneous': 6}
    
    # Spacing between the colormaps of a subplot
    _DC = {'Perceptually Uniform Sequential': 1.4, 'Sequential': 0.7,
           'Sequential (2)': 1.4, 'Diverging': 1.4, 'Cyclic': 1.4,
           'Qualitative': 1.4, 'Miscellaneous': 1.4}
    
    # Indices to step through colormap
    x = np.linspace(0.0, 1.0, 100)
    
    # Do plot
    for cmap_category, cmap_list in cmaps.items():
    
        # Do subplots so that colormaps have enough space.
        # Default is 6 colormaps per subplot.
        dsub = _DSUBS.get(cmap_category, 6)
        nsubplots = int(np.ceil(len(cmap_list) / dsub))
    
        # squeeze=False to handle similarly the case of a single subplot
        fig, axs = plt.subplots(nrows=nsubplots, squeeze=False,
                                figsize=(7, 2.6*nsubplots))
    
        for i, ax in enumerate(axs.flat):
    
            locs = []  # locations for text labels
    
            for j, cmap in enumerate(cmap_list[i*dsub:(i+1)*dsub]):
    
                # Get RGB values for colormap and convert the colormap in
                # CAM02-UCS colorspace.  lab[0, :, 0] is the lightness.
                rgb = mpl.colormaps[cmap](x)[np.newaxis, :, :3]
                lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)
    
                # Plot colormap L values.  Do separately for each category
                # so each plot can be pretty.  To make scatter markers change
                # color along plot:
                # https://stackoverflow.com/q/8202605/
    
                if cmap_category == 'Sequential':
                    # These colormaps all start at high lightness, but we want them
                    # reversed to look nice in the plot, so reverse the order.
                    y_ = lab[0, ::-1, 0]
                    c_ = x[::-1]
                else:
                    y_ = lab[0, :, 0]
                    c_ = x
    
                dc = _DC.get(cmap_category, 1.4)  # cmaps horizontal spacing
                ax.scatter(x + j*dc, y_, c=c_, cmap=cmap, s=300, linewidths=0.0)
    
                # Store locations for colormap labels
                if cmap_category in ('Perceptually Uniform Sequential',
                                     'Sequential'):
                    locs.append(x[-1] + j*dc)
                elif cmap_category in ('Diverging', 'Qualitative', 'Cyclic',
                                       'Miscellaneous', 'Sequential (2)'):
                    locs.append(x[int(x.size/2.)] + j*dc)
    
            # Set up the axis limits:
            #   * the 1st subplot is used as a reference for the x-axis limits
            #   * lightness values goes from 0 to 100 (y-axis limits)
            ax.set_xlim(axs[0, 0].get_xlim())
            ax.set_ylim(0.0, 100.0)
    
            # Set up labels for colormaps
            ax.xaxis.set_ticks_position('top')
            ticker = mpl.ticker.FixedLocator(locs)
            ax.xaxis.set_major_locator(ticker)
            formatter = mpl.ticker.FixedFormatter(cmap_list[i*dsub:(i+1)*dsub])
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_tick_params(rotation=50)
            ax.set_ylabel('Lightness $L^*$', fontsize=12)
    
        ax.set_xlabel(cmap_category + ' colormaps', fontsize=14)
    
        fig.tight_layout(h_pad=0.0, pad=1.5)
        plt.show()