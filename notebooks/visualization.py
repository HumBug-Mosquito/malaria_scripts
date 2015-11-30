# coding: utf-8

from __future__ import division, print_function
import sys
sys.path = ['..'] + sys.path

import os
import numpy as np
from scipy.io import savemat

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from termcolor import colored

from missing_data import replace_missing_values, missing_ratio
import visualizer as viz

ERROR   = "red"
SUCCESS = "green"

### Import Data

# Provide a path to the numpy data and shape files. More shape files can be found at: http://www.gadm.org/download

COUNTRY     = 'cambodia'
ENV         = 'air_temp'
IMPORT_PATH = '/Volumes/DATA/Datasets/Geography_Data/%s/%s/'
SHAPE_PATH  = '/Volumes/DATA/Datasets/Geography_Data/shapefiles/KHM_adm_shp/KHM_adm0.shp'
OUTPUT_PATH = '/Volumes/DATA/Datasets/Geography_Data/viz2'

NORMALISE   = False
GENERATE_MAP= False
SAVE_MAT    = True

CMAP        = {
    'air_temp'   : cm.hot,
    'aet_pet'    : cm.summer,
    'evi'        : cm.BrBG,
    'insolation' : cm.bwr,
    'land_use'   : cm.RdYlGn_r,
    'population' : cm.coolwarm,
}

# Offset the country outline to match up with the raster data:

def offset(x, y):
    offset_line_x = [x - x_offset for x in line_x]
    offset_line_y = [y - y_offset for y in line_y]
    return offset_line_x, offset_line_y


# We need to divide all terms by 10000 to retrieve percentages
def normalise(data):
    for date in data.keys():
        data[date] = data[date] / 10000

    return data

def clip(data, vmin=0, vmax=100):
    for date in data.keys():
        data[date] = np.clip(data[date], vmin, vmax)
    return data

def handle_missing_data(data):
    ### Handle Missing Data
    for date in data.keys():
        missing_data_ratio = missing_ratio(data[date])        
        #Replace missing data with average of neighbours
        if missing_data_ratio:
            data[date] = replace_missing_values(data[date])
            
    return data


def build_evi_map(date, data, no_data, min_, max_, width, height):
    monthly_data = data[date]
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ocean = np.ma.masked_array(monthly_data, monthly_data != no_data)
    
    ocean_map = plt.pcolormesh(ocean, vmin=no_data-0.00001, vmax=no_data+0.00001,
                               alpha=1, cmap=cm.Blues, zorder=10)
    temp_map = plt.pcolor(monthly_data, cmap=mycmap,
                          norm=plt.Normalize(min_, max_))
                          #norm=plt.Normalize(0, 0.8))
    month_label = plt.text(0.95, 0.05, 'date: ' + date,
                           fontsize=20, zorder=20,
                           ha='right', va='bottom', transform=ax.transAxes)

    return (temp_map,ocean_map, month_label) #boundary, 


def generate_map(data, dates):
    print ("generating")
    dpi = 100
    
    # Hardcoded parameters for the visualization:

    x_offset = 3613
    y_offset = 1550
    array_height, array_width = data.values()[0].shape
    print ("coords: ", array_height, array_width)
    viz_width = 8
    aspect_ratio = array_width / array_height
    viz_height = viz_width / aspect_ratio

    
    fig = plt.figure()
    fig.set_size_inches(viz_width, viz_height)
    plt.xlim(xmax=array_width)
    plt.ylim(ymax=array_height)
        
    heatmaps = []
    x = np.arange(-9, 10)
    y = np.arange(-9, 10).reshape(-1, 1)
    base = np.hypot(x,y)
    
    values = sorted(np.unique(data.values()))
    
    no_data = values[0]
    min_    = values[1] # second lowest value
    max_    = values[-1]
    
    print ("range of values:", min_, max_, "[%.4f]" % no_data)
    
    for i, date in enumerate(dates):
        print ("[%d/%d]" % (i, len(dates)))
        heatmaps.append(build_evi_map(date, data, no_data, min_, max_, array_width, array_height))
    
    cbar = plt.colorbar()
    #boundary = plt.plot(offset_line_x, offset_line_y, c='k', linewidth=1, linestyle='-', dash_capstyle='projecting', zorder=5)
    #cbar.set_ticks([0, 1])
    cbar.set_label('Enhanced Vegetation Index')
    plt.gca().invert_yaxis()
    plt.gca().set_title("%s Map for %s, 2000-2014" % (ENV.replace("_", " ").title(), COUNTRY.title()), fontsize=17)
    map_ani = animation.ArtistAnimation(fig, heatmaps, interval=200, repeat=True, repeat_delay=None, blit=True)
    writer = animation.writers['ffmpeg'](fps=10, bitrate=1800)
    plt.tight_layout()
    plt.grid()
    
    outfilename = '%s/%s_%s.mp4' % (OUTPUT_PATH, COUNTRY, ENV)
    map_ani.save(outfilename, dpi=dpi)
    print ("saved in %s" % outfilename)
    
    del map_ani
    del writer
    del cbar
    del outfilename
    del data, date
    del heatmaps
    
    plt.close()
    plt.clf()
    
    import gc
    gc.collect()
    
    #plt.show()


def save_mat(data):
    data2 = {"d%s" % k.replace("_", ""):v for k,v in data.items()}
    savemat('%s/%s-%s-data.mat' % (OUTPUT_PATH, COUNTRY, ENV), data2)
    

def run (country=COUNTRY, env=ENV):
    
    print ("running %s in %s" % (colored(env, SUCCESS), colored(country, SUCCESS)))
    import_path = IMPORT_PATH % (env, country)
    
    print ("loading data")
    # Load the numpy data into memory:
    data = viz.import_numpy_files(import_path)
    
    print ("missing data")
    # Handle missing data
    data = handle_missing_data(data)
    
    
    print ("loading outline")
    # Load country outline:
    (line_x, line_y) = viz.country_outline(SHAPE_PATH)
    
    if env == "evi":
        print ("normalising")
        data = normalise(data)

    print ("sorting")
    dates = sorted([date for date in data.keys()])
    
    font = {'family' : 'Arial',
            'weight' : 'normal',
            'size'   : 16}
    plt.rc('font', **font)
    
    if GENERATE_MAP:
        generate_map(data, dates)
    
    if SAVE_MAT:
        print ("saving mat file")
        save_mat(data)
    
if __name__ == '__main__':

    global mycmap, country
    counter = 1
    
    countries = ["benin", "cambodia", "mozambique", "somalia"]
    envs      = ["evi", "air_temp", "aet_pet", "insolation", "land_use", "population"]
    for COUNTRY in countries:
        for ENV in envs:
            
            print ("[%d/%d]" % (counter, len(countries)*len(envs)))
            mycmap = CMAP[ENV]
            
            try:
                run(COUNTRY, ENV)
            except StopIteration:
                print (colored("NO DATA", ERROR))
            
            counter += 1
            print ()
            
    
