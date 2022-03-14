import pandas as pd
import geopandas as gpd
from os.path import join
from tqdm import tqdm
from glob import glob


RGI = pd.DataFrame()
dir_path = "/home/sa42/data/glac/RGI_TOPO/extracted_outlines/"
full_path = join(dir_path, "*.shp")
for file in tqdm(glob(full_path)):
    glac_reader = gpd.read_file(file)
    RGI = RGI.append(glac_reader, ignore_index = True)
    RGI = RGI.drop("BgnDate", axis=1)
    RGI = RGI.drop("EndDate", axis=1)
    RGI = RGI.drop("O1Region", axis=1)
    RGI = RGI.drop("O2Region", axis=1)
    RGI = RGI.drop("Status", axis=1)
    RGI = RGI.drop("Form", axis=1)
    RGI = RGI.drop("TermType", axis=1)
    RGI = RGI.drop("Surging", axis=1)
    RGI = RGI.drop("Linkages", axis=1)
    RGI = RGI.drop("Name", axis=1)
    RGI = RGI.drop("check_geom", axis=1)
    RGI = RGI.drop("Zmed", axis=1)
    RGI = RGI.drop("Connect", axis=1)
    RGI = RGI.drop("CenLon", axis=1)
    RGI = RGI.drop("CenLat", axis=1)
    RGI = RGI.drop("Area", axis=1)
    RGI = RGI.drop("Zmin", axis=1)
    RGI = RGI.drop("Zmax", axis=1)
    RGI = RGI.drop("Slope", axis=1)
    RGI = RGI.drop("Aspect", axis=1)
    RGI = RGI.drop("Lmax", axis=1)
    RGI = RGI.drop("geometry", axis=1)
    RGI = gpd.GeoDataFrame(RGI)
    RGI.to_csv("~/notebooks/glac/RGI/RGI.csv", index = False)