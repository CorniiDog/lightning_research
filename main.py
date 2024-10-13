import os
from datetime import datetime
import pandas as pd  # Pands for csv creation and output
import matplotlib.pyplot as plt # For 3D plotting
import streamlit as st # Streamlit

st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

from typing import (
    Dict,
    Callable,
    Any,
    Tuple,
    List,
)  # For explicit types to rigid-ify the coding process

import dataParser as dp # dataParser.py

# File tree structure:
#
# project_dir
# -> lightning_data [containing the .dat files] 
# -> -> LYLOUT_240911_155000_0600.dat
# -> -> ...


######################################################################################################
## Folder and Output
######################################################################################################
lightning_data_folder: str = "lightning_data"

# LYLOUT_240911_155000_0600.dat <-
data_extension: str = ".dat"

# The file for finding the cities
dp.cities_file = "ne_110m_populated_places/ne_110m_populated_places.shp"


######################################################################################################
# dataParser.py configuration parameters
#  Note: These configure the parser such that when `dp.parse_file(f, month, day, year)` is ran it
#  appropriately indexes it
######################################################################################################

# Data: time (UT sec of day), lat, lon, alt(m), reduced chi^2, P(dBW), mask <- Identifying the header itself
dp.data_header_startswith = "Data:"

# *** data *** <- Identifying the start of the data
dp.data_body_start = "*** data ***"

# i.e. Data start time: 09/11/24 20:40:00
dp.start_time_indicator = "Data start time:"
dp.start_time_format = "%m/%d/%y %H:%M:%S"

# Configuring latitude, longitude, and altitude
dp.latitude_header = 'lat'
dp.longitude_header = 'lon'
dp.altitude_meters_header = 'alt(m)'

# Callback functions for processing based on header, for translation
#
# Dict[str, Callable[[str], Any]] 
# Basically means we are explicitly defining a dictionary named `process_handling`
# with the key being a string `str`, and the value being a Callable object 
# `Callable[[str], Any]` (basically a function). The callable function must have the 
# parameter be a string but the return value can be anything.
dp.process_handling = {
    "time (UT sec of day)": lambda my_str: float(my_str),  # Convert to float
    "lat": lambda my_str: float(my_str),  # Convert to float
    "lon": lambda my_str: float(my_str),  # Convert to float
    "alt(m)": lambda my_str: float(my_str),  # Convert to float
    "reduced chi^2": lambda my_str: float(my_str),  # Convert to float
    "P(dBW)": lambda my_str: float(my_str), # Convert to float
    "mask": lambda hex_str: int(hex_str, 16) # Convert the hex-code mask to decimal
}

######################################################################################################
# dataParser.py filter procedure to accept data between ranges, and reject rows that don't accept
######################################################################################################

# Sliders for filter parameters
st.sidebar.header("Filter Settings (`-1` = disable)")
# chi_min = st.sidebar.slider("Reduced chi^2 min", 0, 100, 0)
# chi_max = st.sidebar.slider("Reduced chi^2 max", -1, 1000, 50)
# km_min = st.sidebar.slider("Altitude (km) min", 0, 100, 0)
# km_max = st.sidebar.slider("Altitude (km) max", -1, 100, 20)
# mask_count_min = st.sidebar.slider("Mask minimum occurances", 1, 10, 2)

chi_min = st.sidebar.slider("Reduced chi^2 min", 0, 100, 0)
chi_max = st.sidebar.slider("Reduced chi^2 max", -1, 1000, 50)
km_min = st.sidebar.slider("Altitude (km) min", 0, 100, 0)
km_max = st.sidebar.slider("Altitude (km) max", -1, 200, 200)
mask_count_min = st.sidebar.slider("Mask minimum occurances", 1, 10, 2)

do_topography_mapping = st.sidebar.checkbox(label="Enable Topography", value=False)
dp.downsampling_factor = st.sidebar.slider("Topography Downsampling (Size Reduction) Factor", 1, 100, 20)


# Update the dp.filters and dp.count_required dynamically
dp.filters = [
    ["reduced chi^2", lambda chi: chi >= chi_min],
    ["alt(m)", lambda alt: km_max * 1000 >= alt >= km_min * 1000],
]

if chi_max >= 0:
    dp.filters.append(["reduced chi^2", lambda chi: chi <= chi_max])

dp.count_required = [
    ["mask", lambda count: count >= mask_count_min]
]


######################################################################################################
# main function that goes through the files 
######################################################################################################
def main():
    st.title("Lightning Data Parser")
    
    # Get list of .dat files
    dat_files = [f for f in os.listdir(lightning_data_folder) if f.endswith(data_extension)]

    for file in dat_files:
        data_result: pd.DataFrame = dp.get_dataframe(lightning_data_folder, file)


    
        
if __name__ == "__main__":
    main()
