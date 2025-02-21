import os
from typing import List
import numpy as np
import h5py
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import Normalize, BoundaryNorm, LinearSegmentedColormap

from utils.fixedValues import NORM_DICT_TOTAL as normDict

import cartopy
import pyproj
import cartopy.crs as ccrs

rivers = cartopy.feature.NaturalEarthFeature(category='physical', name='rivers_lake_centerlines', scale='10m')
border = cartopy.feature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m')


curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
dat_dir = os.path.dirname(os.path.dirname(curr_dir))
lon = h5py.File(dat_dir+'/data/Auxillary/grid_2d.h5','r')['lon'][:, :]
lat = h5py.File(dat_dir+'/data/Auxillary/grid_2d.h5','r')['lat'][:, :]

def get_cmap_0405(type):  # Adjusted to return a dictionary
    cmap_info = {}
    if type in ['CH4', 'CH5', 'CH6', 'CH7', 'CH8', 'CH9', 'CH10', 'CH11']: #== 'ir':  # SEVIRI channels
        colors1 = plt.cm.plasma(np.linspace(0., 1, 128))
        colors2 = plt.cm.gray_r(np.linspace(0, 1, 128))
        colors = np.vstack((colors1, colors2))
        cmap_info['cmap'] = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        vmin, vmax = 190, 290

    elif type in ['CH1', 'CH2', 'CH3']:
        colors1 = plt.cm.plasma(np.linspace(0., 1, 128))
        colors2 = plt.cm.gray_r(np.linspace(0, 1, 128))
        colors = np.vstack((colors1, colors2))
        cmap_info['cmap'] = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        vmin, vmax = 0, 50       
    elif type == 'LGHT':
        # Generate the Reds color gradient
        reds_colors = plt.cm.copper_r(np.linspace(0, 1, 128))   
        # Prepend white
        colors = np.vstack(([1, 1, 1, 1], reds_colors))  # RGBA for white is [1, 1, 1, 1]
        # Create a new colormap from the extended color gradient
        cmap_info['cmap'] = mcolors.LinearSegmentedColormap.from_list('custom_red', colors)
        vmin, vmax = 0, 4
    elif type in ['INCA prep.', 'Radar prep.', 'CAPE', 'Target', 'EF4INCA', 'imerg', 'precip']: # == 'precip':
        vmin, vmax = 0, 154
        intervals = [vmin, 0.2, 0.3, 0.6, 0.9, 1.7, 2.7, 5, 8.6, 15, 27.3, 50, 89.9, vmax]
        colors = ['#fffffe', '#0101ff', '#0153ff', '#00acfe', '#01feff',
                  '#8cff8c', '#fffe01', '#ff8d01', '#fe4300', '#f60100',
                  '#bc0030', '#ad01ac', '#ff00ff']
        cmap_info['cmap'] = LinearSegmentedColormap.from_list('custom_colormap', colors)
        cmap_info['norm'] = BoundaryNorm(intervals, cmap_info['cmap'].N)
        return cmap_info  # Early return as norm is already defined
    elif type == 'cape':
        vmin, vmax = 0, 300
        cmap_info['cmap'] = 'viridis'
    elif type == 'dem':
        vmin, vmax = 0, 800
        #
        dem_colors = plt.cm.terrain(np.linspace(0.17, 1, 128))   
        # Prepend white
        colors = np.vstack(([1, 1, 1, 1], dem_colors))
        # Create a new colormap from the extended color gradient
        cmap_info['cmap'] = mcolors.LinearSegmentedColormap.from_list('dem_colors', colors)
    elif type == 'lat':
        vmin, vmax = 45, 50
        cmap_info['cmap'] = 'viridis'
    elif type == 'lon':
        vmin, vmax = 8, 18
        cmap_info['cmap'] = 'viridis'

    # For cases without custom norm, define a standard Normalize
    if 'norm' not in cmap_info:
        cmap_info['norm'] = Normalize(vmin=vmin, vmax=vmax)

    return cmap_info

def prepare_data2plot(in_seq, target_seq, pred_seq, normDict,
                      ):
    #### BREAK DOWN THE DATA!
    dem = in_seq[-7, :, :, -1]
    seviri = in_seq[:, :, :, 0:-1] # This would still work even if there is only one sevir channel!
    ## Time to DEnormalize the data!
    seviri[:, :, :, -1] = (seviri[:, :, :, -1] * normDict['ch11']['std'] + normDict['ch11']['mean'])
    seviri[:, :, :, -2] = (seviri[:, :, :, -2] * normDict['ch10']['std'] + normDict['ch10']['mean'])
    seviri[:, :, :, -3] = (seviri[:, :, :, -3] * normDict['ch09']['std'] + normDict['ch09']['mean'])
    seviri[:, :, :, -4] = (seviri[:, :, :, -4] * normDict['ch08']['std'] + normDict['ch08']['mean'])
    seviri[:, :, :, -5] = (seviri[:, :, :, -5] * normDict['ch07']['std'] + normDict['ch07']['mean'])
    seviri[:, :, :, -6] = (seviri[:, :, :, -6] * normDict['ch06']['std'] + normDict['ch06']['mean'])
    seviri[:, :, :, -7] = (seviri[:, :, :, -7] * normDict['ch05']['std'] + normDict['ch05']['mean'])
    seviri[:, :, :, -8] = (seviri[:, :, :, -8] * normDict['ch04']['std'] + normDict['ch04']['mean'])
    seviri[:, :, :, -9] = (seviri[:, :, :, -9] * normDict['ch03']['std'] + normDict['ch03']['mean'])
    seviri[:, :, :, -10] = (seviri[:, :, :, -10] * normDict['ch02']['std'] + normDict['ch02']['mean'])
    seviri[:, :, :, -11] = (seviri[:, :, :, -11] * normDict['ch01']['std'] + normDict['ch01']['mean'])

    dem = (dem * normDict['dem']['std'] + normDict['dem']['mean'])
    target_seq = np.power(10, target_seq)
    pred_seq = np.power(10, pred_seq)
    return seviri, dem, target_seq, pred_seq
   
def plot_layer(ax, data, cmap_info, title=None, fs=10, show_cbar=False, cbar_label=None, row=None, col=None, plotting_ir=False, plotting_vis=False, add_borders=True, add_rivers=False):
    """
    Enhanced function to plot a single layer with dynamic row and column handling.
    """
    if row is not None and col is not None:
        axis = ax[row, col]
    else:
        axis = ax

    # Ensure axis has a projection set, default to PlateCarree if not specified
    if axis.projection is None:
        axis.projection = ccrs.PlateCarree()


    im = axis.imshow(np.flipud(data), 
                     extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                     transform=ccrs.PlateCarree(),
                     **cmap_info)
    
    if title:
        axis.set_title(title, fontsize=fs)
    axis.xaxis.set_ticks([])
    axis.yaxis.set_ticks([])
    axis.set_aspect(248/184, adjustable='box')
    # Optionally add country borders
    if add_borders:    
        axis.add_feature(cartopy.feature.BORDERS, edgecolor='black')
        axis.add_feature(cartopy.feature.COASTLINE, linewidth=1, color='black')
    if add_rivers:
        # axis.add_feature(rivers, edgecolor='blue')
        axis.add_feature(cartopy.feature.RIVERS, edgecolor='blue')
    if show_cbar:
        # Create a new axis for the colorbar on the left of the plot
        cbar = plt.colorbar(im, ax=axis, location='left', orientation='vertical', fraction=0.035, pad=0.04)
        if cbar_label:
            cbar.set_label(cbar_label, size=fs*.8)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        if plotting_ir:
            cbar.set_ticks([200, 220, 240, 260, 280])
        cbar.ax.tick_params(labelsize=fs*.4)
        if plotting_vis:
            cbar.set_ticks([10, 20, 30, 40, 50])
        cbar.ax.tick_params(labelsize=fs*.4)
        
    return im

def plot_past_regulars(ax, seviri, time_indices, time_points, cmap_dict, fs):
    """
    Plot SEVIRI channels dynamically based on the number of available bands.
    Channels are dropped in the order of 5, 6, and then 7, with channel 9 always present.
    """
    # Define the mapping for SEVIRI channels based on the input shape
    # channel_mapping = {
    #     4: [5, 6, 7, 9]  # If 4 bands, they are channels 5, 6, 7, and 9
    # }
    channel_mapping = {
        11: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    }
    # Determine the available channels based on the number of SEVIRI bands
    num_bands = seviri.shape[-1]  # Assuming last dimension is channels
    
    channels = channel_mapping[num_bands]
    # Loop through the available channels and plot
    for row, channel in enumerate(channels):
        for col, time_idx in enumerate(time_indices):
            if row in [0, 1, 2]:
                plot_layer(ax[row, col], seviri[time_idx, :, :, row], cmap_dict(f'CH{channel}'), show_cbar=(col==0), cbar_label=f'CH {channel}', plotting_ir=False, plotting_vis = True, fs=fs)
            else:
               plot_layer(ax[row, col], seviri[time_idx, :, :, row], cmap_dict(f'CH{channel}'), show_cbar=(col==0), cbar_label=f'CH {channel}', plotting_ir=True, plotting_vis = False, fs=fs) 

    for col, t_trans_idx in enumerate(time_indices):
        # Add the timestamps now!
        ax[0][col].set_title('{} Mins'.format(time_points[col]), fontsize=fs)
    return row + 1 # Return the row index for the next plot

def plot_future(ax, target_seq, pred_seq, time_indices, time_points, cmap_dict, row2work, fs):
    """
    Plot the future target and predicted sequences.
    """
    for col, t_trans_idx in enumerate(time_indices):
        plot_layer(ax[row2work, col], target_seq[t_trans_idx, :, :], cmap_dict('Target'), show_cbar=(col==0), cbar_label='Target', fs=fs)
        plot_layer(ax[row2work+1, col], pred_seq[t_trans_idx, :, :], cmap_dict('precip'), show_cbar=(col==0), cbar_label='Output', fs=fs)
        ax[row2work, col].set_title('{} Mins'.format(time_points[col]), fontsize=fs)

def plot_incaData(in_seq, target_seq, pred_seq, titleStr=None,
                               norm=normDict, 
                               figsize=(15, 20), # (w, h)
                               fs=10,
                               dpi=300, 
                               save_path=None,
                               savePdf=False,
                               ):
    """
    Refactored plotting function for INCA data.
    """
    # Data preparation
    seviri, dem, \
        target_seq, pred_seq = prepare_data2plot(in_seq, target_seq, pred_seq, norm)
    # Set the colormap dictionary
    cmap_dict = lambda s: get_cmap_0405(s)
    nrows = seviri.shape[-1] + 6  # Additional rows for lght, inca_precip, radar, aux, target, and pred
    ncols = 7
    if pred_seq.shape[0] == 24:
        time_points_past = [-120, -90, -60, -30, -15, -5, 0]
        time_indices_past = [  0,   6,  12,  18,  21, 23, 24]
        #
        time_points_future = [5, 10, 15, 30, 60, 90, 120]
        time_indices_future = [0, 1,  2,  5, 11, 17, 23]
    elif pred_seq.shape[0] == 18:
        time_points_past =  [-90, -60, -30, -15, -10, -5, 0]
        time_indices_past = [  0,   6,  12,  15,  16, 17, 18]
        #
        time_points_future = [5, 10, 15, 30, 45, 60, 90]
        time_indices_future =[0,  1,  2,  5,  8, 11, 17]
    elif pred_seq.shape[0] == 12:
        time_points_past = [-60, -45, -30, -20, -10, -5, 0]
        time_indices_past =  [0,   3,   6,   8,  10, 11, 12]
        #
        time_points_future = [5, 10, 15, 20, 30, 45, 60]
        time_indices_future =[0,  1,  2,  3,  5,  8, 11]
        #
    elif pred_seq.shape[0] == 1:
        ncols = 9
        time_points_past = [-120, -105, -90, -75, -60, -45, -30, -15, 0]
        time_indices_past =  [0,   1,   2,   3,  4, 5, 6, 7, 8]
        #
        time_points_future = [0]
        time_indices_future =[0]
    #
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, layout='constrained', subplot_kw={'projection': ccrs.PlateCarree()})
    # print("Type of ax:", type(ax))
    # if isinstance(ax, np.ndarray):
    #     print("Shape of ax array:", ax.shape)

    
    ## First part is input data w/o aux!
    row2work = plot_past_regulars(ax, seviri, time_indices_past, time_points_past, cmap_dict, fs=fs)
    # Row of aux data will be plotted manually!
    if seviri.shape[0] == 25:
        col4aux = 2
    else: # This can be generic since only 2h lead time uses two capes among the 3 lead time options!
        col4aux = 1
    # Turn for DEM #### Don't plot lat and lon!, LAT, and LON
    plot_layer(ax[row2work, col4aux], dem, cmap_dict('dem'), title='DEM', fs=fs) # , show_cbar=True
    # Rest is empty!
    for i in range(col4aux+1, 8):
        ax[row2work, i].axis('off')    
    #### Time to go to the future!
    row2work += 1
    plot_future(ax, target_seq, pred_seq, time_indices_future, time_points_future, cmap_dict, row2work, fs)
    #
    if titleStr:
        fig.suptitle(titleStr, fontsize=fs*1.5)
    #
    if save_path:
        if savePdf:
            fig.savefig(save_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
        else:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def save_example_vis_results(
        save_dir, save_prefix, in_seq, target_seq, pred_seq, label,
        fs=24, norm=None, title=None,
        savePdf=False
        ):
    in_seq = in_seq[0,:,:,:,:].astype(np.float32)
    target_seq = target_seq[0,:,:,:,0].astype(np.float32)
    pred_seq = pred_seq[0,:,:,:,0].astype(np.float32)
    timestamp = title[0].split('_')[-1][0:11]
    fig_path = save_dir +'/'+ save_prefix + '_' + timestamp + '.png'

    if savePdf: # We also need to pimp up the title in this case!
        timestamp = f'{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[8:10]}:{timestamp[10:12]}'
        title = f'Sample at t = {timestamp}'
        fig_path = fig_path.replace('.png', '.pdf')
    else:
        title = title[0].split('/')[-1][:-3]

    plot_incaData(in_seq=in_seq, target_seq=target_seq, pred_seq=pred_seq, 
        titleStr=title, 
        save_path=fig_path,
        savePdf=savePdf, 
        fs=fs, 
        figsize=(20, 15)
        )