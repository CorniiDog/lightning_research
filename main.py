import os
from datetime import datetime
import pandas as pd  # Pands for csv creation and output
import matplotlib.pyplot as plt # For 3D plotting
import streamlit as st # Streamlit

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
# ->
# -> lightning_data_output [empty, will be outputted with .csv files after main.py runs]
# -> -> LYLOUT_240911_155000_0600.csv
# -> -> ... 


######################################################################################################
## Folder and Output
######################################################################################################
lightning_data_folder: str = "lightning_data"

# LYLOUT_240911_155000_0600.dat <-
data_extension: str = ".dat"

######################################################################################################
# dataParser.py configuration parameters
#  Note: These configure the parser such that when `dp.parse_file(f, month, day, year)` is ran it
#  appropriately indexes it
######################################################################################################

# Data: time (UT sec of day), lat, lon, alt(m), reduced chi^2, P(dBW), mask <- Identifying the header itself
dp.data_header_startswith = "Data:"

# *** data *** <- Identifying the start of the data
dp.data_body_start = "*** data ***"

######################################################################################################
# dataParser.py parameter procedure to translate the read string to more appreciable types
######################################################################################################
def str_to_float(my_str: str) -> float:
    """
    Converts a string to a float\n
    :param my_str: The string (i.e. `"3.12"`)
    :return: The float representing the hex (`3.12`)
    """
    return float(my_str)


def str_hex_to_int(hex_str: str) -> int:
    """
    Converts a hexcode (i.e. `0x1721`) to an int (`5921`)\n
    :param hex_str: The hex string (i.e. `"0x1721"`)
    :return: An integer representing the hex (`5921`)
    """
    return int(hex_str, 16)


# Callback functions for processing based on header, for translation
#
# Dict[str, Callable[[str], Any]] 
# Basically means we are explicitly defining a dictionary named `process_handling`
# with the key being a string `str`, and the value being a Callable object 
# `Callable[[str], Any]` (basically a function). The callable function must have the 
# parameter be a string but the return value can be anything.
dp.process_handling = {
    "time (UT sec of day)": str_to_float,
    "lat": str_to_float,
    "lon": str_to_float,
    "alt(m)": str_to_float,
    "reduced chi^2": str_to_float,
    "P(dBW)": str_to_float,
    "mask": str_hex_to_int
}

######################################################################################################
# dataParser.py filter procedure to accept data between ranges, and reject rows that don't acept
######################################################################################################
def accept_chi_below(chi: float) -> bool:
    return chi <= 50.0


def data_above_20_km(alt_meters: float) -> bool:
    km = alt_meters / 1000.0  # Convert to kilometers
    return km >= 20.0


# After conversion process, you can now add callback functions for filters
# Accepts the designated row if the function returns true
#
# List[str, Callable]
# We are explicitly defining a list (or array) to the variable `filters`
dp.filters = [
    ["reduced chi^2", accept_chi_below],
    ["alt(m)", data_above_20_km]
]

######################################################################################################
# dataParser.py count procedure to accept data such that there must be a given
# number of occurances to be accepted (1, 2, 3, etc..?)
######################################################################################################
def mask_count_rule(n: int) -> bool:
    return n > 1

# This determines count instances rule
# That is, if you provide a header, you provide a function for the count of items
# For example, for 1 >= n >= 2 that requires instances to be between 1 and 2 occurances
dp.count_required = [
    ["mask", mask_count_rule]
]


######################################################################################################
# main function that goes through the files 
######################################################################################################
def main():
    st.title("Lightning Data Parser")
    
    # Get list of .dat files
    dat_files = [f for f in os.listdir(lightning_data_folder) if f.endswith(data_extension)]
    
    # Streamlit file selector
    selected_file = st.selectbox("Select a data file:", dat_files)
    
    if selected_file:
        # Parse the selected file
        data_result: pd.DataFrame = dp.get_dataframe(lightning_data_folder, selected_file)

        # Display the parsed DataFrame
        st.write("Parsed Data:")
        st.dataframe(data_result)
        
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
