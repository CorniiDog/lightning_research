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
# Topography data
# Can find respective lat and long to use here: https://www.latlong.net/
######################################################################################################
dp.params["south"] = 27.458283  # Modify these coordinates for your area of interest (Texas)
dp.params["north"] = 33.481568
dp.params["west"] = -101.810989
dp.params["east"] = -91.924764

######################################################################################################
# dataParser.py filter procedure to accept data between ranges, and reject rows that don't accept
######################################################################################################

# Sliders for filter parameters
st.sidebar.header("Filter Settings (`-1` = disable)")
chi_min = st.sidebar.slider("Reduced chi^2 min", 0, 100, 0)
chi_max = st.sidebar.slider("Reduced chi^2 max", -1, 1000, 50)
km_min = st.sidebar.slider("Altitude (km) min", 0, 100, 20)
km_max = st.sidebar.slider("Altitude (km) max", -1, 1000, -1)
mask_count_min = st.sidebar.slider("Mask minimum occurances", 1, 10, 2)

# Update the dp.filters and dp.count_required dynamically
dp.filters = [
    ["reduced chi^2", lambda chi: chi >= chi_min],
    ["alt(m)", lambda alt: alt >= km_min * 1000],
]

if chi_max >= 0:
    dp.filters.append(["reduced chi^2", lambda chi: chi <= chi_max])

if km_max > 0:
    dp.filters.append(["alt(m)", lambda alt: alt <= km_max * 1000])

dp.count_required = [
    ["mask", lambda count: count >= mask_count_min]
]


######################################################################################################
# main function that goes through the files 
######################################################################################################
def main():
    st.title("Lightning Data Parser")

    # Section: File Upload
    st.header("Upload a `.dat` file")
    uploaded_file = st.file_uploader("", type="dat")
    
    if uploaded_file is not None:
        save_path = save_uploaded_file(uploaded_file)
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    st.divider()

    # Section: File Management
    st.header("Manage `.dat` files")
    
    # Get list of .dat files
    dat_files = [f for f in os.listdir(lightning_data_folder) if f.endswith(data_extension)]
    
    # Streamlit file selector
    selected_file = st.selectbox("Select a data file:", dat_files)
    
    if selected_file:

        col1, col2 = st.columns(2)

        # Parse the selected file
        data_result: pd.DataFrame = dp.get_dataframe(lightning_data_folder, selected_file)

        # Plot 3D scatter
        fig = dp.plot_interactive_3d(data_result, 'mask')

        # Display the 3D plot in Streamlit
        col1.plotly_chart(fig)

        # Display the parsed DataFrame
        col2.write("Parsed Data:")
        col2.dataframe(dp.color_df(data_result, 'mask'))
        
        # Option to download the DataFrame as a CSV
        csv_output = data_result.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_output,
            file_name=f"{os.path.splitext(selected_file)[0]}.csv",
            mime="text/csv"
        )

        # Option to delete the selected file
        if st.button(f"Delete {selected_file}"):
            if delete_file(selected_file):
                st.success(f"File '{selected_file}' deleted successfully!")
            else:
                st.error(f"Error: Could not delete file '{selected_file}'.")


######################################################################################################
# Helper functions for file management
######################################################################################################
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(lightning_data_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def delete_file(file_name):
    file_path = os.path.join(lightning_data_folder, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False
        
if __name__ == "__main__":
    main()
