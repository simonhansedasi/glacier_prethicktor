# glacier_prethicktor
A machine learning approach to predicting glacier thicknesses.

---

## Model Assembly 

---

**The Glacier Thickness Predictor (GTP) is divided into four modules representing different datasets used in model training: sm, sm2, sm4, sm5.**



**sm:** \
training data - glacier \
avg test mae range: 36.024810 - 99.422656 \
avg train mae range: 10.851193 - 35.202624

**sm2:** \
training data - Glam \
avg test mae range: 10.680168 - 16.546986 \
avg train mae range: 9.754352 - 16.536474 

**sm4:** \
training data - Glam_phys \
avg test mae range: 14.431750 - 31.223658 \
avg train mae range: 12.829470 - 31.172126

**sm5:** \
training data - Glam_2 \
avg test mae range: 44.656052 - 69.933537 \
avg train mae range: 27.547083 - 71.829759 





---

## User Guide

---
1. Prepare model architecture:
    - Update glacierml.py with desired layer architecture
        1. modify function build_and_run_model()
            1. update 'arch' variable with architecture 'N-N-N'
            2. update svd_mod_pth and svd_res_pth for dataset in use
                - sm = glacier
                - sm2 = Glam
                - sm4 = Glam_phys
                - sm5 = Glam_2
        3. modify layer structure in function build_dnn_model() and comment out any unneeded layers.
        
2. Prepare prethicktor.py with hyperparameters and data

    - prethicktor.py will load all data by default, but will need data directory inputs to the loader functions.
    
    - Hyperparameters available to adjust are learning rate, validation split, and random state. By default validation split is left at 0.2 to avoid inundation of different models:
        1. learning rates - 0.1, 0.01, 0.001
        2. validation split - 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4
        3. random state - range from 0 to 24
               
    - Function build_and_train_model requires several inputs:
        1. dataset - no need to modify. This input is set when module is selected at the beginning.
        2. module - no need to modify. This input is set when module is selected at the beginning. Determines which dataset will be used and where to save the models and results.
        3. res - no need to modify. This input is set when module is selected at the beginning.
        4. learning_rate - defined earlier as range LR. Default is 0.001
        5. validation_split - defined earlier as range VS. Default is 0.2
        6. epochs - default is 300
        7. random_state - defined earlier as a range 0 to 24. Default is 0
        

4. start prethicktor container and run 
    - here you may do thing a
    - or you may do thing b

5. run results_builder.py

6. view results in zults_grabber.ipynb


---

## Files and Operations

---
**This GTP consists of three python files and an interactive notebook.**


1.  **glacierml.py** \
Python file setting the following functions: 

    data_loader(path to data)
    - This functions input requires a directory path to where data is located. It is intended to load updated GlaThiDa glacier.csv data hosted on [GitLab](https://gitlab.com/wgms/glathida/-/tree/main/data) (download required). Older versions of the function currently commented into legacy code are intended to load the older versions of the GlaThiDa data: [website](https://www.gtn-g.ch/glathida/), [zip download](http://www.gtn-g.ch/database/glathida-3.1.0.zip)       
        
    data_loader_2(path to data)
    - updated version of data_loader that builds Glam using thicknesses from T data matched with RGI features. The glaciers are matched using some tools located in RGI_tools/
    
    thickness_renamer(dataset)
    - Discrepancies exist between the name of thickness columns of GlaThiDa datasets. This function renames columns of T (glacier) and TT (band) datasets from 'mean_thickness' to 'thickness' to match TTT(point) dataset, and for consistency in results.   
    
    data_splitter(dataset, random_state)
    - This function defines test and train datasets as well as features vs labels for a given random state. 

    build_linear_model(normalizer, learning_rate)
    - description
    
    plot_loss(history)
    - dfdsa
    
    build_and_train_model(dataset, learning_rate, validation_split, epochs, random_state)
    - fdsa
        
   
2.  **prethicktor.py** \
 The main function file. Intended to run in a docker container on a GPU.
 

3.  **results_builder.py** \
scripts to evaluate models and make predictions of RGI data using selected model


4.  **zults_grabber.ipynb** \
interactive notebook used to analyze results

---

## Datasets and their Assembly

---
### glacier
This dataset is simply the glacier dataset from GlaThiDa hosted on [GitLab](https://gitlab.com/wgms/glathida/-/tree/main/data) (download required)


### Glam
Glam is built out of GlaThiDa glacier thicknesses combined with RGI glacier attributes. Each glacier in GlaThiDa is matched with a glacier in RGI using geopy.distance. If the distance between the glaciers is less than 1 km, then both index of GlaThiDa and RGI are saved to a csv file which is located with other data files.


    for T_idx in tqdm(T.index):
        GlaThiDa_coords = (T['LAT'].loc[T_idx],
                           T['LON'].loc[T_idx])
        for RGI_idx in RGI_coordinates.index:
            RGI_coords = (RGI_coordinates['CenLat'].loc[RGI_idx],
                          RGI_coordinates['CenLon'].loc[RGI_idx])
            distance = geopy.distance.geodesic(GlaThiDa_coords, RGI_coords).km
            
            
The gl.data_loader_2 function reads uses the index csv and drops glaciers where distance is non-zero, as well as any RGI duplicates in case GlaThiDa centroid is equidistant from multiple RGI glaciers. 


    comb = pd.read_csv(pth + 'GlaThiDa_RGI_matched_indexes.csv')
    drops = comb.index[comb['0']!=0]
    comb = comb.drop(drops)
    comb = comb.drop_duplicates(subset = 'RGI_index', keep = 'last')
            
            
At this point, GlaThiDa and RGI data are selected from what remains of valid matched GlaThiDa and RGI indexes, and then have their indexes reset. This is done so that GlaThiDa and RGI data line up, but have matching indexes for the merge into Glam:


    T = T.loc[comb['GlaThiDa_index']]
    T = T.reset_index()
    RGI = RGI_extra.loc[comb['RGI_index']]
    RGI = RGI.reset_index()
    
    
Once the indexes match and the data is lined up, it is a simple merge to put them together:


    Glam = pd.merge(T, RGI, left_index=True, right_index=True)


### Glam_phys
Glam data without centroid lat and lon


    Glam_phys = Glam[[
    #     'CenLon',
    #     'CenLat',
        'Area',
        'thickness',
        'Slope',
        'Zmin',
        'Zmed',
        'Zmax',
        'Aspect',
        'Lmax'
    ]]


### Glam_2
Key difference between Glam and Glam_2: Glam is built with GlaThiDa 'T' dataset, while Glam_2 is built using the updated glacier dataset hosted on GitLab.

Rebuild of Glam using different techniques of matching. When building Glam, data_loader_2 would produce a list of matched indexes with several duplicates for unknown buggy reasons. With data_loader_3 to build Glam_2, some more care and rigor was put into the matcher with the intent of having a more accurate dataset. For each match that was to be put into a list, an index locator was integrated to ensure only the last entry of the dataframe was updated. This stopped the problem of several duplicates being saved at once, which occured when building Glam.

Another key difference in Glam_2 index matching is that each GlaThiDa entry is compared to every single RGI entry, as opposed to the previous version would break the loop once it found a match less than 1 km away. This loop keeps every single index within 1 km to be analyzed later.


        distance = geopy.distance.geodesic(GlaThiDa_coords,RGI_coords).km
        if 0 <= distance < 1:
            f = pd.Series(distance, name='distance')
            L = L.append(f, ignore_index=True)
            L['GlaThiDa_index'].iloc[-1] = T_idx
            L['RGI_index'].iloc[-1] = RGI_idx
            L.to_csv('GlaThiDa_RGI_live.csv')
            
            
            
Once the matched indexes are collected, there are several potential matches since this loop did not break after it found the first match within 1km. The next step is to find the closest RGI glacier to each GlaThiDa glacier. This is done with the following loop:


    combined_indexes = pd.DataFrame()
    for GlaThiDa_index in comb['GlaThiDa_index'].index:
        df = comb[comb['GlaThiDa_index'] == GlaThiDa_index]
        f = df.loc[df[df['distance'] == df['distance'].min()].index]
        combined_indexes = combined_indexes.append(f)
        
        
        
Now that GlaThiDa and closest RGI indexes are matched, it is time to match GlaThiDa thicknesses with RGI attributes.


    data = pd.DataFrame(columns = ['GlaThiDa_index', 'thickness'])
    for GlaThiDa in combined_indexes['GlaThiDa_index'].index:
        glathida_thickness = glacier['mean_thickness'].iloc[GlaThiDa] 
        rgi_index = combined_indexes['RGI_index'].loc[GlaThiDa]  
        rgi = RGI_extra.iloc[[rgi_index]]

        data = data.append(rgi)
        data['thickness'].iloc[-1] = glathida_thickness
