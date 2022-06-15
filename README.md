# glacier_prethicktor
A machine learning approach to predicting glacier thicknesses.

---

## Use Guide

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
        
2. Prepare prethicktor.py with hyperparameters
    - Modify prethicktor.py for selected data and hyperparameters
        1. fdsa
        2. fdsa
        3. fdsa
        4. fdsa

4. start prethicktor container and run 
    - here you may do thing a
    - or you may do thing b

5. run results_builder.py

6. view results in zults_grabber.ipynb

7. profit

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

## Data Sets and Assembly

---
glacier


Glam


Glam_phys


Glam_2




---

## Model Assembly 

---

**The Glacier Thickness Predictor (GTP) is divided into four training approaches: sm, sm2, sm4, sm5.**



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
