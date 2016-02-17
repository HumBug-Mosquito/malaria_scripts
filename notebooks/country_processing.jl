# coding: utf-8

## Benin Data Processing

using HDF5
using MAT

ROOT = "/Volumes/DATA/Datasets/GeoData/"
FORMAT = (7200, 3000)

# Define a bounding box for each country (in degrees of lat/lon):

BBOX = Dict(
    "benin" =>
        Dict(
            "top_left_lat"=> 12.5,
            "top_left_lon"=> 0.65,
            "width"=> 3.25, 
            "height"=> 6.4,
        ),
    "kenya" =>
        Dict(
            "top_left_lat"=> 4.5,
            "top_left_lon"=> 33.5,
            "width"=> 8.5,
            "height"=> 9.5,
        )
)

function convert_bbox(coords, limits=FORMAT)
    """
    Convert the bounding box from coordinates to indices of the data provided.
    """
    
    xmax, ymax = limits
    println(coords)
    width  = coords["width"]
    height = coords["height"]
    
    #left   = floor(Int, (coords["top_left_lon"] + 180) * xmax / 360. )
    #right  = ceil(Int, (coords["top_left_lon"]+width + 180) * xmax / 360. )
    #top    = ceil(Int, (coords["top_left_lat"] + 60) * ymax / 180.)
    #bottom = floor(Int, (coords["top_left_lat"]-height + 60) * ymax / 180.)

    left   = floor(Int, (coords["top_left_lon"] + 180) * xmax / 360. )
    right  = ceil(Int, (coords["top_left_lon"]+width + 180) * xmax / 360. )
    top    = ceil(Int, (90 - coords["top_left_lat"]) * ymax / (90.+60))
    bottom = floor(Int, (90 - coords["top_left_lat"]+height) * ymax / (90.+60))

    
    #print ("coords:\n  ", coords, "\ncorrespond to indices:\n  %d->%d, %d->%d" %(left, right, bottom, top))

    return left, right, top, bottom
end

#function save_mat(data, path)
#    data2 = {"d%s" % k.replace("_", ""):v for k,v in data.items()}
#    savemat("%s-data.mat" % path, data2)


for datum in ["aet_pet"] #["evi", "temp", "ins"] # "aet_pet"
    
    if datum == "aet_pet"
        reshape_format = (FORMAT[2], FORMAT[1])
    else
        reshape_format = FORMAT
    end
    
    print_with_color(:blue, datum*"\n")
    
    for country in ["kenya"]
        
        print_with_color(:blue, country*"\n")
        
        ### Import data
       
        IMPORT_PATH = joinpath(ROOT, "uncompressed", datum)
        println(IMPORT_PATH)
        #EXPORT_PATH = "%s/%s/%s" % (ROOT, datum, country)
        EXPORT_PATH = "/Volumes/DATA/Datasets/Geography_Data/viz2/$country/$datum"
        mkpath(EXPORT_PATH)
        hd5_filename = "$EXPORT_PATH/$datum.md5"
        mat_filename = "$EXPORT_PATH/$datum.mat"
        left, right, top, bottom = convert_bbox(BBOX[country], FORMAT)
        
        println("Exporting to $EXPORT_PATH")
        close(h5open(hd5_filename, "w"))
        
        months_data = Dict()
        for file_ in readdir(IMPORT_PATH)
            
            if !endswith(file_,".txt")
                continue
            end
            
            println("  processing ", file_)
            month = "d"*replace(basename(file_)[1:7], "_", "")
            
            data = reshape(readdlm(joinpath(IMPORT_PATH, file_)), reshape_format)
            
            if datum == "aet_pet"
                data = data'
            end
            
            #month_data = rotl90(data[ left:right, top:bottom])
            month_data = data[ left:right, top:bottom]'
            months_data[month] = month_data
            
            h5open("$EXPORT_PATH.h5", "r+") do file
                write(file, month, month_data)  # alternatively, say "@write file A"
            end

            file = matopen(mat_filename, true, true, true, false, false)
            write(file, month, month_data)
            close(file)
            
        end    
        
        #output = EXPORT_PATH   
        #np.save("%s.npy" % output, months_data)
        #save_mat(months_data, "%s.mat" % output)
        
        
    end    
end
