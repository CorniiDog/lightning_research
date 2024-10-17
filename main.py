import os
import pandas as pd
import dataParser as dp # dataParser.py
import streamlit as st
import numpy as np
from datetime import datetime
from streamlit_timeline import st_timeline
from dateutil.relativedelta import relativedelta
from typing import (
    Dict,
    Callable,
    Any,
    Tuple,
    List,
)  # For explicit types to rigid-ify the coding process

st.set_page_config(
    page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None
)

st.markdown(f'''
    <style>
        section[data-testid="stSidebar"] .css-ng1t4o {{width: 14rem;}}
        section[data-testid="stSidebar"] .css-1d391kg {{width: 14rem;}}
    </style>
''',unsafe_allow_html=True)

st.title("Connor White's Lightning Data Parser")
st.divider()

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

# Get list of .dat files
dat_files = [f for f in os.listdir(path=lightning_data_folder) if f.endswith(data_extension)]

# dataParser.py filter procedure to accept data between ranges, and reject rows that don't accept
# Sliders for filter parameters
st.sidebar.header("Parameters")


with st.sidebar.expander("File/Date Select", expanded=True):


    choices = ["Filter By Date Range", "Filter By Files"]
    filter_type = st.selectbox("Select Filter Type", choices)

    approved_files = []
    start_date = None
    end_date = None
    if filter_type == "Filter By Date Range":
            now: datetime = datetime.now()
            start_date = st.date_input("Start Date", now - relativedelta(months=1))
            dp.start_datetime = datetime(start_date.year, start_date.month, start_date.day)
            end_date = st.date_input("End Date", now)
            dp.end_datetime = datetime(end_date.year, end_date.month, end_date.day)
            approved_files = dat_files
    else:
        dp.start_datetime = None
        start_date = None
        dp.end_datetime = None
        end_date = None

        approved_files = st.multiselect("Select files", dat_files)

with st.sidebar.expander("Filtering Parameters", expanded=True):
    chi_min: int = st.number_input("Reduced chi^2 min", 0, 100, 0)
    chi_max: int = st.number_input("Reduced chi^2 max", 0, 1000, 50)
    km_min: int = st.number_input("Altitude min (km)", 0, 100, 0)
    km_max: int = st.number_input("Altitude max (km)", 0, 200, 200)
    mask_count_min: int = st.slider("Mask minimum occurances", 1, 10, 2)

with st.sidebar.expander("Lightning Parameters", expanded=True):
    dp.lightning_max_strike_time = st.number_input("Lightning maximum allowed strike time between points (s)", 0.0, 2.0, 0.15)
    dp.lightning_max_strike_distance = st.number_input("Lightning maximum allowed strike distance between points (km)", 0.0, 100.0, 3.0) * 1000.0
    dp.lightning_minimum_speed = st.number_input("Lightning minimum allowed speed between points (m/s)", 0.0, 299792458.0, 299792.458)
    dp.min_points_for_lightning = mask_count_min

with st.sidebar.expander(label="Calendar Parameters", expanded=True):
    max_calendar_items: int = st.number_input(
        "Maximum Lightning Strikes To Display", 1, 10000, value=1000
    )
with st.sidebar.expander("Topography Parameters", expanded=True):
    do_topography_mapping: int = st.checkbox(label="Enable Topography", value=False)
    dp.downsampling_factor = st.number_input(
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

    lightning_strikes: List[pd.DataFrame] = []
    strike_times: List[Tuple[str, str]] = []
    
    # Cache files for processing the month
    for file in approved_files:
        if len(approved_files) > 0 and file not in approved_files:
            continue

        with st.spinner(f"Retreiving data for `{file}`"):
            data_result: pd.DataFrame = dp.get_dataframe(lightning_data_folder, file)

        with st.spinner(f"Parsing lightning data for `{file}`"):
            sub_strikes, substrike_times = dp.get_strikes(data_result)
            lightning_strikes += sub_strikes # Concatenate to lightning_strikes
            strike_times += substrike_times


    if len(lightning_strikes) > 0:
        with st.spinner(f"Establishing timeline"):
            items = []
            for i in range(len(lightning_strikes)):
                timeline_start = strike_times[i][0].split("T")
                data_dict = {
                    "id": i,
                    "content": f"{lightning_strikes[i]['mask'][0]} @ {timeline_start[1]}",
                    "start": strike_times[i][0],
                }
                if i > max_calendar_items-1:
                    st.warning(f"Only displaying maximum of {max_calendar_items} items")
                    break
                items.append(data_dict)

        options = {"cluster": True, "snap": True, "stack": False}
        if start_date and end_date:
            options["min"] = start_date.strftime('%Y-%m-%d')
            options["max"] = end_date.strftime('%Y-%m-%d')

        st.header("Select Lightning Event")

        timeline = st_timeline(items, groups=[], options=options, height="150px")
        print(timeline)
        if timeline:
            index: int = timeline["id"]
            data_result: pd.DataFrame = lightning_strikes[index]
            timeline_start = timeline["start"].split("T")
            mask = data_result['mask'][0]
            st.header(f"Lightning strike for mask `{mask}` on `{timeline_start[0]}` at `{timeline_start[1]}` UTC")

            col1, col2 = st.columns(2)

            # Display the parsed DataFrame
            col2.write("Parsed Data:")
            col2.dataframe(dp.color_df(data_result, 'mask'))
            
            # Option to download the DataFrame as a CSV
            csv_output = data_result.to_csv(index=False)
            col2.download_button(
                label="Download CSV",
                data=csv_output,
                file_name=f"{mask}_{timeline_start[0]}_{timeline_start[1]}.csv",
                mime="text/csv"
            )

            # Displaying figure
            fig = None
            with st.spinner('Indexing Topography Data...'):
                # Plot 3D scatter
                fig = dp.get_interactive_3d_figure(data_result, 'mask', do_topography=do_topography_mapping)


            # Display the 3D plot in Streamlit
            if fig:
                col1.plotly_chart(fig)
    else: # No lightning data
        st.warning("Data too restrained. Modify parameters on left sidebar.")

    st.divider()

    with st.expander("Manage `.dat` Files", expanded=False):
        # Section: File Upload
        st.header("Upload a `.dat` File")
        uploaded_file = st.file_uploader("", type="dat")
        
        if uploaded_file is not None:
            save_uploaded_file(uploaded_file)
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")

        st.divider()

        st.header("Delete a `.dat` File")
        
        # Streamlit file selector
        selected_file = st.selectbox("Select a data file:", dat_files)

        # Option to delete the selected file
        if st.button(f"Delete `{selected_file}`"):
            st.error("Do you really want to delete this file?")
            if st.button(f"Yes, delete `{selected_file}`"):
                if delete_file(selected_file):
                    st.success(f"File '{selected_file}' deleted successfully!")
                else:
                    st.error(f"Error: Could not delete file '{selected_file}'.")


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