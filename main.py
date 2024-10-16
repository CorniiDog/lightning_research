import os
import pandas as pd
import dataParser as dp # dataParser.py
import streamlit as st
import numpy as np

st.set_page_config(
    page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None
)

# The lightning data folder containing the .dat files
lightning_data_folder: str = "lightning_data"
# LYLOUT_240911_155000_0600.dat <-
data_extension: str = ".dat"
# The file for finding the cities
dp.cities_file = "ne_110m_populated_places/ne_110m_populated_places.shp"
# dataParser.py configuration parameters
# Note: These configure the parser such that when `dp.parse_file(f, month, day, year)` is ran it
#  appropriately indexes it
# Data: time (UT sec of day), lat, lon, alt(m), reduced chi^2, P(dBW), mask <- Identifying the header itself
dp.data_header_startswith = "Data:"
# *** data *** <- Identifying the start of the data
dp.data_body_start = "*** data ***"
# i.e. Data start time: 09/11/24 20:40:00
dp.start_time_indicator = "Data start time:"
dp.start_time_format = "%m/%d/%y %H:%M:%S"
# Configuring latitude, longitude, and altitude
dp.latitude_header = "lat"
dp.longitude_header = "lon"
dp.altitude_meters_header = "alt(m)"

# Callback functions for processing based on header, for translation
# Basically means we are explicitly defining a dictionary named `process_handling`
# with the key being a string `str`, and the value being a Callable object
# `Callable[[str], Any]` (basically a function). The callable function must have the
# parameter be a string but the return value can be anything.
dp.process_handling = {
    "time (UT sec of day)": float,  # Convert to float
    "lat": float,  # Convert to float
    "lon": float,  # Convert to float
    "alt(m)": float,  # Convert to float
    "reduced chi^2": float,  # Convert to float
    "P(dBW)": float,  # Convert to float
    "mask": str,
}
# dataParser.py filter procedure to accept data between ranges, and reject rows that don't accept
# Sliders for filter parameters
st.sidebar.header("Filter Settings (`-1` = disable)")

chi_min: int = st.sidebar.slider("Reduced chi^2 min", 0, 100, 0)
chi_max: int = st.sidebar.slider("Reduced chi^2 max", -1, 1000, 50)
km_min: int = st.sidebar.slider("Altitude (km) min", 0, 100, 0)
km_max: int = st.sidebar.slider("Altitude (km) max", -1, 200, 200)
mask_count_min: int = st.sidebar.slider("Mask minimum occurances", 1, 10, 2)

do_topography_mapping: int = st.sidebar.checkbox(label="Enable Topography", value=False)
dp.downsampling_factor = st.sidebar.slider(
    "Topography Downsampling (Size Reduction) Factor", 1, 100, 20
)

# Update the dp.filters and dp.count_required dynamically
dp.filters = [
    ["reduced chi^2", lambda chi: chi >= chi_min],
    ["alt(m)", lambda alt: km_max * 1000 >= alt >= km_min * 1000],
]

if chi_max >= 0:
    dp.filters.append(["reduced chi^2", lambda chi: chi <= chi_max])

# This is a filtering for the count of a given component
# I.e. there must be two instances of the same mask to be accepted, therefore
# you can put ["mask", lambda count: count >= 2]
dp.count_required = [["mask", lambda count: count >= mask_count_min]]
# main function that goes through the files
def main():
    st.title("Lightning Data Parser")

    # Get list of .dat files
    dat_files = [f for f in os.listdir(lightning_data_folder) if f.endswith(data_extension)]

    # Cache files for processing the month
    with st.spinner("Parsing data files"):
        for file in dat_files:
            data_result: pd.DataFrame = dp.get_dataframe(lightning_data_folder, file)

            # Assuming 'mask' is the name of the column containing the mask values in your DataFrame
            unique_masks = data_result['mask'].unique()

            # Iterate through each unique mask value
            for mask_value in unique_masks:
                # Filter the DataFrame to get only rows with the current mask value
                mask_subset = data_result[data_result['mask'] == mask_value]

                print(file, mask_subset)
                
                for i in range(len(mask_subset)):
                    for j in range(i + 1, len(mask_subset)):
                        row1 = mask_subset.iloc[i]
                        r1x, r1y, r1z = row1['x(m)'], row1['y(m)'], row1['z(m)']
                        t1 = row1['time (UT sec of day)']

                        row2 = mask_subset.iloc[j]
                        r2x, r2y, r2z = row2['x(m)'], row2['y(m)'], row2['z(m)']
                        t2 = row2['time (UT sec of day)'] # seconds

                        dist = np.sqrt(np.pow(r1x-r2x, 2) + np.pow(r1y-r2y, 2) + np.pow(r1z-r2z, 2))
                        delta_t = t1 - t2
                        potential_speed = np.abs(dist/delta_t)

                        speed_of_light = 299792458 # m/s

                        # margin = 0.001
                        # if potential_speed > speed_of_light * margin and potential_speed < speed_of_light:
                        #     print(potential_speed, row1['mask'])

                        # Run file
                        # See the mask is 3e28 with 3802 rows in LYLOUT_240911_184000_0600_Exported.dat
                        # The mask is spanned by around a 10 minute gap
                break
                    






            

    with st.spinner("Establishing timelines"):
        print("Hello world")


if __name__ == "__main__":
    main()