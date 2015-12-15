# coding: utf-8

## Benin Data Processing

from __future__ import division, print_function
import sys, os
from os.path import join as j, basename as b
import numpy as np
import glob
import termcolor

COL_INFO = 'blue'
COL_ERR  = 'red'
COL_SUCC = 'green'

#sys.path = ['..'] + sys.path
#import country_window as cwp
#from time import sleep

from scipy.io import savemat

ROOT = '/Volumes/DATA/Datasets/GeoData/'
FORMAT = (7200,3000)

# Define a bounding box for each country (in degrees of lat/lon):

BBOX = {
    'benin' :
        {
            'top_left_lat': 12.5,
            'top_left_lon': 0.65,
            'width': 3.25, 
            'height': 6.4,
        },
    'kenya' :
        {
            'top_left_lat': 4.5,
            'top_left_lon': 33.5,
            'width': 8.5,
            'height': 9.5,
        }
}

def convert_bbox(coords, limits=FORMAT):
    """
    Convert the bounding box from coordinates to indices of the data provided.
    """
    
    xmax, ymax = limits
    width  = coords['width']
    height = coords['height']
    
    left   = int(np.floor((coords['top_left_lon'] + 180) * xmax / 360. ))
    right  = int(np.ceil ((coords['top_left_lon']+width + 180) * xmax / 360. ))
    top    = int(np.ceil ((coords['top_left_lat'] + 90) * ymax / 180.))
    bottom = int(np.floor((coords['top_left_lat']-height + 90) * ymax / 180.))
    
    #print ("coords:\n  ", coords, "\ncorrespond to indices:\n  %d->%d, %d->%d" %(left, right, bottom, top))

    return left, right, top, bottom

def save_mat(data, path):
    data2 = {"d%s" % k.replace("_", ""):v for k,v in data.items()}
    savemat('%s-data.mat' % path, data2)

for datum in ['aet_pet']:
    
    print (colored(datum, COL_INFO))
    
    for country in ['kenya']:
        
        print (colored(country, COL_INFO))
        
        ### Import data
       
        IMPORT_PATH = j(ROOT, 'uncompressed', datum)
        print (IMPORT_PATH)
        #EXPORT_PATH = '%s/%s/%s' % (ROOT, datum, country)
        EXPORT_PATH = '%s/%s-%s' % ("/Volumes/DATA/Datasets/Geography_Data/viz2", country, datum)
        left, right, top, bottom = convert_bbox(BBOX[country], FORMAT)
        
        months_data = {}
        for file_ in os.listdir(IMPORT_PATH):
            
            if not file_.endswith(".txt"): continue
            
            print ("  processing", file_)
            month = b(file_)[:7]
            
            data = np.loadtxt(j(IMPORT_PATH, file_)).reshape( FORMAT )
            months_data[month] = data[ bottom:top, left:right]
            
        
        print(months_data.keys())
        output = EXPORT_PATH   
        np.save("%s.npy" % output, months_data)
        save_mat(months_data, "%s.mat" % output)
        
        