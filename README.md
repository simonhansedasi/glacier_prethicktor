# glacier_prethicktor
A machine learning approach to predicting glacier thicknesses.

We treat estimating glacier thickness as a regression problem as opposed to ice velocity flow models. We train machine learning models with Glacier Thickness Database thickness measurements as an independent variable co-registered with surface attributes from the Randolph Glacier Inventory as dependent variables to estimate thickness.
## Requirements
<ul>
    <li> tensorflow v. 2.8.0
    <li> 
</ul>


## Data
<ul>
    <li> Download RGI v. 6.0 [here](https://www.glims.org/RGI/rgi60_dl.html).
    <li> Download GlaThiDa 3.1.0 [here](https://www.gtn-g.ch/database/glathida-3.1.0.zip).
</ul>

### Conda environment
```conda create -n glacierml python=3.7 pandas=1.3.5 scikit-learn numpy yellowbrick plotly tensorflow geopy```
