# glacier_prethicktor
A machine learning approach to predicting glacier thicknesses.

### Table of Contents:

#### 1. Project Description
#### 2. Detailed description
#### 3. Workflow
#### 4. Data
#### 5. Module Details


---

## 1. Project Description

---
<p>
Knowledge of the total volume of glacier ice on Earth is an important benchmark for understanding and adapting to our changing climate. Several estimates of global glacier ice volume have recently been presented (Farinotti et al., 2019; Milan et al., 2022). These previous estimates have relied on simple, physics-based models of glacier flow. Here, we examine whether an entirely data-driven estimate of ice mass is possible.
</p>

<p>
We train a neural network on thickness measurements from the Glacier Thickness Database (GlaThiDa). We use a simple shallow/fat architecture (two dense layers and several times more neurons than input variables). Dropout layers are added to reduce the tendency to overfit the data. We treat the learning rate, number of training epochs, and the number of neurons per dense layer as tunable hyperparameters. We perform bootstrap aggregating wherein an ensemble of randomly seeded models are trained and averaged to produce one thickness estimate. We then evaluate the ensemble on the entire Randolph Glacier Inventory (RGI) with the result being a global estimate of non-ice sheet glacier volume.
</p>

<p>
The Glacier Thickness Predictor (GTP) consists of four python files, an example workflow notebook, and 5 more notebooks for model and data analysis. The example notebook provides a simplified version of the model workflow, while the main GTP is run through a terminal window and is capable of running in a docker container on either CPU or GPU. A detailed workflow is described in part 3.
</p>

---

## 2. Detailed description

---
<ol>

<li> <b> glacierml.py </b> </li>

<p>
This file contains all the functions used throughout the GTP. Imported as gl.
</p>

<li> <b> model_builder.py </b></li>

<p>
This file contains scripts to build and train ML models to predict the thickness of glaciers. When run the user will be prompted to select a module. These modules represent different ways of assembling training data with gl.data_loader() and are detailed later in part 5.
</p>

<p>
After a training module is selected, the user is then prompted for layer architecture, learning rate, and epochs. These hyperparameters are useful knobs to tweak to improve model performance, but as a first run on a module, the defaults used in this project are:
</p>

<ul>
<li> layer 1 = 10 </li>
<li> layer 2 = 5 </li>
<li> learning rate = 0.01 </li>
<li> epochs = 100 </li>
</ul>

<br>

<p>
With the hyperparameters input, the model_builder.py will build and train two ensembles of models. The first ensemble includes a dropout layer and the second ensemble does not include a dropout layer. The models and their histories are then saved in their respective saved folders in the projects home directory. The models and histories can also be saved into memory as a variable, demonstrated in the example-workflow notebook.
</p>

<li> <b> results_builder.py </b> </li>
<p>
The results_builder.py file loads and evaluates the models in a given training module and then saves the results to a csv. The CLI in the terminal will prompt for a module when run.
</p>

<p>
results_builder.py will load and evaluate all models in a selected module using gl.predictions_maker(). This function assembles a dataframe of model parameters and predictions made on training and testing datasets, identifiable by the given random state of selected data. Each thickness is then multiplied by the area used in its prediction to compute a predicted volume. Volumes are then summed and divided by the summed area of the dataset to produce an average thickness across all predicted glaciers in a given train or test dataset. This process is repeated for all 25 random states in the ensemble. After all models have been evaluated, results_builder.py will save a .csv file of all model predictions.
</p>

<p>
These predictions tables are then passed to the gl.deviations_calculator() function to compute the standard deviations and variances across the ensembles. This function will collapse each 25 entry predictions table into a single row for a deviations table showing the average value and standard deviations for both model mean absolute error and predicted thicknesses. These deviations tables are then loaded in the ML analysis notebook to examine predictions and loss curves, as well read by prethicktor.py to make global or regional predictions for glaciers in the Randolph Glacier Inventory.
</p>

<li> <b> prethicktor.py </b> </li>
<p>
This python file loads a desired model ensemble to make predictions for glaciers with unknown thicknesses in the Randolph Glacier Inventory (RGI).
</p>

<p>
When run, prethicktor.py first prompts for a training module selection. Then a table will be displayed to allow the user to select a model ensemble from a combination of layer architecture, dropout selection, learning rate, and epochs. The selected ensemble will load 25 models which predict 25 unique thickness, which are then averaged providing a mean thickness and variance. These mean thicknesses and variances are saved, alongside the feature data used to make predictions, as a .csv file in the 'zults/' project folder.
</p>

<li> <b> example_workflow.ipynb </b> </li>
<p>
example_workflow.ipynb provides a simplified version of the GTP workflow. It is designed as a tutorial for the project and will complete a full workflow of a single model, not an ensemble. Models and results are saved in a module not accesseble to the following python files: model_builder.py, results_builder.py, and prethicktor.py.
</p>

<li> <b> ml_analysis.ipynb </b> </li>
<p>
ml_analysis.ipynb is used to analyze performance of GTP models. The first two cells allow for module selection and model parameter calculation. These calculated parameters form the layer architecture used to tune model performance.
</p>
<p>
Next, the notebook contains cells to load a 'deviations' table. This table contains model ensemble information such as inputs, parameters, layer architecture, learning rate, and epochs, as well as statistics from model performance such as test and train MAE and predicted thickness averages and standard deviations. One of these ensembles is chosen in the next cells to evaluate the model ensemble. A cross-plot is generated of predictions made on the training and test data set combined, as well as a cross-plot of the ensemble loss curves.
</p>
<li> <b> vol_comp.ipynb </b> </li>
<p>
vol_comp.ipynb is a notebook used to generate plots comparing GTP predicted thicknesses to reference thicknesses published in <a href = 'https://rdcu.be/cT84m'> Farinotti et. al 2019  A consensus estimate for the ice thickness distribution of all glaciers on Earth </a>.
</p>

<li> <b> clusters.ipynb </b> </li>
<p>
clusters.ipynb is used for cluster analysis on different RGI statistics.
</p>
</ol>




---

## 3. Workflow

---

### Step 1:
Assemble or select a module of training data \
![Image](figs/readme/data_selection.png)


### Step 2:
Calculate layer architecture using zults grabber notebook. \
![Image](figs/readme/parameter_calculator.png)


### Step 3:
Run python file model_builder.py for desired module. The CLI will ask for layer architecture, learning rate, and epochs. \

![Image](figs/readme/model_builder.png)


### Step 4:
Run python file results_builder.py for desired module. \
![Image](figs/readme/results_builder.png)


### Step 5:
Analyze ML results in zults grabber notebook and change parameters as needed. The notebook will load all models that have results, and it is possible to select which data to view. \
![Image](figs/readme/deviations_analysis.png)


### Step 6:
Run python file prethicktor.py on selected module. A table of trained models will load, select one to use for making thickness predictions. \
![Image](figs/readme/prethicktor_1.png)


Once selected, the GTP will predict thicknesses and calcuate deviations across the 25 models for each region \
![Image](figs/readme/prethicktor_part_2.png)
### Step 7:
Load predicted thicknesses for desired model in zults grabber notebook and analyze results


---

## 4. Data

---


---

## 5. Module Details

---




### sm1

GlaThiDa data only, no RGI attributes added.

total inputs: 440

avg test mae range: 34.423079 - 89.075339 \
avg train mae range: 12.263926 - 14.530515


**Layer Architecture 10-5** -- *Default*\
total model parameters: 120 \
trained model parameters: 111 \
optimized learning rate: 0.1 \
optimized epochs: \
sum total predicted volume: \
sum total predicted volume variance:


**Layer Architecture 16-8** -- *1/2 parameters to inputs* \
total model parameters: 234 \
trained model parameters: 225 \
optimized learning rate: \
optimized epochs: \
sum total predicted volume: \
sum total predicted volume variance:


**Layer Architecture 20-15** -- *1/1 parameters to inputs* \
total model parameters: 440 \
trained model parameters: 431 \
optimized learning rate: \
optimized epochs: \
sum total predicted volume: \
sum total predicted volume variance:


**Layer Architecture 35-22** -- *experimental overparameterization* \
total model parameters: 999 \
trained model parameters: 990 \
optimized learning rate: \
optimized epochs: \
sum total predicted volume: \
sum total predicted volume variance:



### sm2

GlaThiDa thickness data merged with RGI physical attributes. No size comparison between GlaThiDa and RGI glaciers applied, i.e. area_scrubber = off

total inputs: 3834

avg test mae range: 27.289853 - 31.172687 \
avg train mae range: 19.661094 - 24.033518

**Layer Architecture 10-5** -- *Default*\
total model parameters: 180 \
trained model parameters: 161 \
optimized learning rate: \
optimized epochs: \
sum total predicted volume: \
sum total predicted volume variance:


**Layer Architecture 50-28** -- *1/2 parameters to inputs* \
total model parameters: 1976 \
trained model parameters: 1957 \
optimized learning rate: \
optimized epochs: \
sum total predicted volume: \
sum total predicted volume variance:


**Layer Architecture 64-48** -- *1/1 parameters to inputs* \
total model parameters: 3828 \
trained model parameters: 3809 \
optimized learning rate: \
optimized epochs: \
sum total predicted volume: \
sum total predicted volume variance:



### sm3

GlaThiDa thickness data merged with RGI physical attributes. Sizes of RGI and GlaThiDa glaciers compared and only kept in training data if the difference in size is less than 1 km sq.


total inputs: 2304


avg test mae range: 12.414377 - 21.948432 \
avg train mae range: 11.061126 - 22.97135

**Layer Architecture 10-5** -- *Default*\
total model parameters: 180 \
trained model parameters: 161 \
optimized learning rate: 0.1 \
optimized epochs: 10 \
sum total predicted volume:  \
sum total predicted volume variance:


**Layer Architecture 35-22** -- *1/2 parameters to inputs* \
total model parameters: 1184 \
trained model parameters: 1165 \
optimized learning rate: \
optimized epochs: \
sum total predicted volume: \
sum total predicted volume variance:


**Layer Architecture 55-30** -- *1/1 parameters to inputs* \
total model parameters: 2280 \
trained model parameters: 2261 \
optimized learning rate: \
optimized epochs: \
sum total predicted volume: \
sum total predicted volume variance:




### sm4

GlaThiDa thickness data merged with RGI physical attributes. Sizes of RGI and GlaThiDa glaciers compared and only kept in training data if the difference in size is less than 5 km sq.

total inputs: 3015

avg test mae range: 14.268922 - 15.341705 \
avg train mae range: 11.86620 - 14.398697

**Layer Architecture 10-5** -- *Default*\
total model parameters: 180 \
trained model parameters: 161 \
optimized learning rate: \
optimized epochs: \
sum total predicted volume: \
sum total predicted volume variance:


**Layer Architecture 48-21** -- *1/2 parameters to inputs* \
total model parameters: 1550 \
trained model parameters: 1531 \
optimized learning rate: \
optimized epochs: \
sum total predicted volume: \
sum total predicted volume variance:


**Layer Architecture 60-39** -- *1/1 parameters to inputs* \
total model parameters: 3038 \
trained model parameters: 3019 \
optimized learning rate: \
optimized epochs: \
sum total predicted volume: \
sum total predicted volume variance:



<ul>
<br>
<li> <b> sm6 </b> </li>
GlaThiDa thicknesses with GlaThiDa features only (Area, Mean Slope, Centroid Latitude, Centroid Longitude)
<br>
    <br>
<li> <b> sm2 </b> </li>
GlaThiDa thickness data combined with RGI surface features on a global scale. No corrections for size anomalies (Area mismatch between GlaThiDa and RGI)
<br>
    <br>


<li> <b> sm3 </b> </li>
GlaThiDa thickness data combined with RGI surface features on a global scale. Corrections for size anomalies include dropping glaciers with size difference greater than 1 km
<br>
    <br>
<li> <b> sm4 </b> </li>
GlaThiDa thickness data combined with RGI surface features on a global scale. Corrections for size anomalies include dropping glaciers with size difference greater than 5 km
<br>
    <br>

<li> <b> sm5 </b> </li>
GlaThiDa thickness data combined with RGI surface features on a global scale. No corrections for size anomalies (Area mismatch between GlaThiDa and RGI). Dropped data column 'Zmed' from training and predictions as it contains several erroneous data.
<br>
    <br>

<li> <b> sm6 </b> </li>
GlaThiDa thickness data combined with RGI surface features on a regional scale. No correction for size anomalies. Each data set is trained and predicted for only that RGI region
<br>
    <br>

<li> <b> sm7 </b></li>
GlaThiDa thickness data combined with RGI surface features on a global scale. No corrections for size anomalies (Area mismatch between GlaThiDa and RGI). **NOTE** sm7 is the same as sm2, however, at the time the prethicktor.py file was not set up to make predictions regionally. To make regional predictions, an entire new module was required. prethicktor.py has since been patched such that any module can predict regionally.
<br>
    <br>

<li> <b> sm8 </b> </li>
GlaThiDa thickness data combined with RGI surface features on a global scale. No corrections for size anomalies. Dropped data column 'Zmed' from training and predictions as it contains several erroneous data.
</ul>
