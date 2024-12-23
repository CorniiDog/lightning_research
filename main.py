import os, io
from datetime import datetime
from typing import List, Tuple
import dataParser as dp # dataParser.py
import numpy as np
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta
from streamlit_timeline import st_timeline
import imageio.v3 as iio
import time
import pytz

st.set_page_config(
    page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None
)

st.markdown(
    f'''
    <style>
        section[data-testid="stSidebar"] .css-ng1t4o {{width: 14rem;}}
        section[data-testid="stSidebar"] .css-1d391kg {{width: 14rem;}}
    </style>
''',
    unsafe_allow_html=True,
)


st.title("Connor White's Lightning Data Parser")

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


def main():

    dp.polish_data_cache(lightning_data_folder=lightning_data_folder)

    with st.sidebar.expander("Manage `.dat` Files", expanded=False):
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
                    dp.polish_data_cache(lightning_data_folder=lightning_data_folder)
                    st.rerun()

                else:
                    st.error(f"Error: Could not delete file '{selected_file}'.")

    with st.sidebar.expander("File/Date Select", expanded=True):

        formatted_files = []
        for i in range(len(dat_files)):
            file = dat_files[i]
            formatted_files.append(
                f"{dp.get_start_date_label(lightning_data_folder, file)}: {file}"
            )

        selected_files = st.multiselect("Select files", formatted_files)

        for i in range(len(selected_files)):
            selected_file: str = selected_files[i]
            selected_files[i] = selected_file.split(": ")[-1]

        now: datetime = datetime.now()

        st.divider()

        start_date = st.date_input("Start Date", now - relativedelta(months=1))
        start_t = st.time_input("Start Time (UTC)", value=datetime.min.time())  # default to 00:00

        st.divider()

        end_date = st.date_input("End Date", now)
        end_t = st.time_input("End Time (UTC)", value=datetime.min.time())      # default to 00:00

        # Combine date and time
        start_datetime = datetime(
            start_date.year, 
            start_date.month, 
            start_date.day, 
            start_t.hour, 
            start_t.minute, 
            tzinfo=pytz.UTC
        )
        end_datetime = datetime(
            end_date.year, 
            end_date.month, 
            end_date.day, 
            end_t.hour, 
            end_t.minute, 
            tzinfo=pytz.UTC
        )

        approved_files = selected_files
        if len(selected_files) == 0:
            approved_files = dat_files

    with st.sidebar.expander("Filtering Parameters", expanded=True):
        chi_min: int = st.number_input("Reduced chi^2 min", 0, 100, 0)
        chi_max: int = st.number_input("Reduced chi^2 max", 0, 1000, 50)
        km_min: int = st.number_input("Altitude min (km)", 0, 100, 0)
        km_max: int = st.number_input("Altitude max (km)", 0, 200, 200)
        mask_count_min: int = st.slider("Mask minimum occurances", 2, 10, 4)

    with st.sidebar.expander("Lightning Parameters", expanded=True):
        lightning_max_strike_time = st.number_input(
            "Lightning maximum allowed strike time between points (s)", 0.0, 2.0, 0.15
        )
        lightning_max_strike_distance = (
            st.number_input(
                "Lightning maximum allowed strike distance between points (km)", 0.0, 200.0, 50.0
            )
            * 1000.0
        )
        lightning_minimum_speed = st.number_input(
            "Lightning minimum allowed speed between points (m/s)", 0.0, 299792458.0, 299792.458
        )
        min_points_for_lightning = mask_count_min

    with st.sidebar.expander("Topography Parameters", expanded=True):
        do_topography_mapping: int = st.checkbox(label="Enable Topography", value=False)
        buffer_factor = st.number_input("Topography Overlap Size (lat/lon)", 0.0, 2.0, 0.1)

    with st.sidebar.expander("Graph Parameters", expanded=True):
        dp.interactive_3d_dot_size = st.slider("Dot Size", 1, 15, 5)
        dp.interactive_2d_dot_size = int(dp.interactive_3d_dot_size * (8/5))
        dp.dot_size_min = st.slider("Dot Minimum Size", 0,  15, 3)

    # Update the dp.filters and dp.count_required dynamically
    filters = [
        ["reduced chi^2", lambda chi: chi >= chi_min],
        ["alt(m)", lambda alt: km_max * 1000 >= alt >= km_min * 1000],
    ]

    if chi_max >= 0:
        filters.append(["reduced chi^2", lambda chi: chi <= chi_max])

    # This is a filtering for the count of a given component
    # I.e. there must be two instances of the same mask to be accepted, therefore
    # you can put ["mask", lambda count: count >= 2]
    count_required = [["mask", lambda count: count >= mask_count_min]]
    # main function that goes through the files

    lightning_strikes: List[pd.DataFrame] = []
    strike_times: List[Tuple[str, str]] = []

    # Cache files for processing the month
    len_approved_files = len(approved_files)
    file_progress_bar = st.progress(0, text=f"Adressing Relevant Files:")
    for i, file in enumerate(approved_files):
        if len_approved_files > 0 and file not in approved_files: # Basically if there IS a selection for files
            continue
            
        if len_approved_files > 0:
            file_progress_bar.progress(value=(i+1)/len_approved_files, text=f"Adressing Relevant Files: {(100*(i+1)/len_approved_files):.1f}%")

        with st.spinner(f"Retreiving data for `{file}`"):
            data_result: pd.DataFrame = dp.get_dataframe(
                lightning_data_folder=lightning_data_folder,
                file_name=file,
                count_required=count_required,
                filters=filters,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
            )

        if data_result is not None and len(data_result) > 0:
            with st.spinner(f"Parsing lightning data for `{file}`"):
                sub_strikes, substrike_times = dp.get_strikes(
                    df=data_result,
                    lightning_max_strike_time=lightning_max_strike_time,
                    lightning_max_strike_distance=lightning_max_strike_distance,
                    lightning_minimum_speed=lightning_minimum_speed,
                    min_points_for_lightning=min_points_for_lightning,
                )

                if len(substrike_times) > 0:
                    lightning_strikes += sub_strikes  # Concatenate to lightning_strikes
                    strike_times += substrike_times
    file_progress_bar.empty()

    if len(lightning_strikes) > 0:

        with st.expander("Select Lightning Event", expanded=True):
            
            timeline_tab, selection_tab = st.tabs(tabs=["Select From Timeline", "Select Individually"])
            
            with timeline_tab:
                max_calendar_items: int = st.number_input(
                    "Maximum Lightning Strikes To Display", 1, 10000, value=1500
                )

                with st.spinner(f"Establishing timeline"):
                    items = []
                    for i in range(len(lightning_strikes)):
                        if len(lightning_strikes[i]) < min_points_for_lightning:
                            continue
                        # Convert strike_times[i][0] to a datetime, then make ISO 8601 string with "Z"
                        dt_utc = datetime.strptime(strike_times[i][0], "%Y-%m-%dT%H:%M:%S")
                        dt_utc = dt_utc.replace(tzinfo=pytz.UTC)
                        iso_utc_str = dt_utc.isoformat().replace("+00:00", "Z")

                        data_dict = {
                            "id": i,
                            "content": f"{lightning_strikes[i]['mask'][0]} {dt_utc.strftime('%H:%M:%S')} ({len(lightning_strikes[i])} pts)",
                            "start": iso_utc_str,  # Full date+time in UTC
                        }
                        if i > max_calendar_items - 1:
                            st.warning(f"Only displaying maximum of {max_calendar_items} items")
                            break
                        items.append(data_dict)

                options = {"cluster": False, "snap": False, "stack": False}
                if start_date and end_date:
                    options["min"] = start_datetime.isoformat().replace("+00:00", "Z")
                    options["max"] = end_datetime.isoformat().replace("+00:00", "Z")

                timeline = st_timeline(items, groups=[], options=options, height="150px")

                if timeline:
                    fine_tune_seconds: int = st.slider("Fine-tune seconds", 1, 100, 5)

                    strike_found = True
                    unix_time = datetime.strptime(timeline["start"], "%Y-%m-%dT%H:%M:%SZ").timestamp()
                    index: int = timeline["id"]

                    index_lookup = {timeline["content"]: index}
                    for i in range(len(strike_times)):
                        if i == index:
                            continue

                        strike_time = datetime.strptime(
                            strike_times[i][0], "%Y-%m-%dT%H:%M:%S"
                        ).timestamp()
                        if np.abs(unix_time - strike_time) < fine_tune_seconds:

                            timeline_start = strike_times[i][0].split("T")
                            name = f"{lightning_strikes[i]['mask'][0]} {timeline_start[1]} ({len(lightning_strikes[i])} pts)"
                            index_lookup[name] = i

                    strike_name: str = st.selectbox(
                        f"Fine-tune selection within `{fine_tune_seconds}` seconds",
                        list(index_lookup.keys()),
                    )
                    index: int = index_lookup[strike_name]
                    timeline_start = strike_times[index][0].split("T")
                    data_result: pd.DataFrame = lightning_strikes[index]
                    mask = data_result["mask"][0]
                else:
                    strike_found = False

            with selection_tab:
                sort_options = ["Points", "Date", "Mask"]
                sort_method = st.selectbox("Sort events by:", sort_options)

                # Build a list of events, each containing all info needed
                events = []
                for i in range(len(lightning_strikes)):
                    if len(lightning_strikes[i]) < min_points_for_lightning:
                        continue

                    event_mask = lightning_strikes[i]["mask"].iloc[0]
                    event_time_str = strike_times[i][0]  # e.g. "2024-09-11T15:50:00"
                    dt = datetime.strptime(event_time_str, "%Y-%m-%dT%H:%M:%S")
                    pts_count = len(lightning_strikes[i])

                    events.append({
                        "index": i,
                        "mask": event_mask,
                        "datetime": dt,
                        "pts": pts_count,
                    })

                # Sort events based on user’s choice
                if sort_method == "Date":
                    # Sort by date ascending, then pts ascending
                    events.sort(key=lambda x: (x["datetime"], x["pts"]))
                elif sort_method == "Mask":
                    # Sort by mask ascending, then date ascending
                    events.sort(key=lambda x: (x["mask"], x["datetime"]))
                else:
                    # Default: sort by points ascending
                    events.sort(key=lambda x: x["pts"])

                # Build the dictionary for selectbox
                event_selection_map = {}
                for ev in events:
                    user_friendly_time = ev["datetime"].strftime("%Y-%m-%d %H:%M:%S")
                    label = f"{ev['mask']} [{user_friendly_time}] ({ev['pts']} pts)"
                    event_selection_map[label] = ev["index"]

                if event_selection_map:
                    chosen_label = st.selectbox("Choose an event", list(event_selection_map.keys()))
                    chosen_index = event_selection_map[chosen_label]

                    data_result = lightning_strikes[chosen_index]
                    timeline_start = strike_times[chosen_index][0].split("T")
                    mask = data_result["mask"].iloc[0]
                    strike_found = True
                else:
                    st.warning("No lightning events found for selection.")
                    strike_found = False


        if strike_found:
            st.header(
                f"Lightning strike for mask `{mask}` on `{timeline_start[0]}` at `{timeline_start[1]}` UTC"
            )

            col1, col2 = st.columns(2)

            # Display the parsed DataFrame
            col2.write("Parsed Data:")
            col2.dataframe(dp.color_df(data_result, "mask"))


            col2col1, col2col2 = col2.columns(2)

            # Option to download the DataFrame as a CSV
            csv_output = data_result.to_csv(index=False)
            col2col1.download_button(
                label="Download CSV",
                data=csv_output,
                file_name=f"{mask}_{timeline_start[0]}_{timeline_start[1]}.csv",
                mime="text/csv",
            )

            excel_output = io.BytesIO()
            data_result.to_excel(excel_output, index=False)
            excel_output.seek(0)
            col2col2.download_button(
                label="Download Excel",
                data=excel_output,
                file_name=f"{mask}_{timeline_start[0]}_{timeline_start[1]}.xlsx",
                mime="xlsx",
            )


            # Displaying figure
            fig = None
            with st.spinner('Creating 3D Figure Data'):
                # Plot 3D scatter
                fig = dp.get_interactive_3d_figure(
                    data_result, "mask", buffer_factor, do_topography=do_topography_mapping
                )

            with st.spinner("Plotting 3D Figure"):
                # Display the 3D plot in Streamlit
                col1.plotly_chart(fig)

            with st.spinner("Creating 2D Figure Data"):
                lonalt_fig = dp.get_3_axis_plot(
                    data_result, "mask", buffer_factor, do_topography=do_topography_mapping
                )

            with st.spinner("Plitting 2D Figure"):
                st.plotly_chart(lonalt_fig)

            with st.expander("Generate `.gif` Image", expanded=False):
                max_length = st.number_input("Maximum gif length (s)", 0.5, 10.0, 5.0)
                fps = st.number_input("Frames Per Second (fps)", 1, 60, value=10)

                repeat_gif = st.checkbox("Repeat Gif", value=True)
                if do_topography_mapping:
                    use_full_resolution = not st.checkbox("Use Full Resolution Topography (Expensive)", value=False)
                else:
                    use_full_resolution = False
                
                total_frames = int(max_length * fps)

                if st.button("Generate"):
                    with st.spinner("Generating gif"):

                        progress_text = "Generating Images:"
                        my_bar = st.progress(0, text=progress_text)

                        frames = []

                        len_data_result = len(data_result)
                        steps = np.linspace(0, 1, total_frames)  # Steps from 0 to 1
                        indices = (steps * (len_data_result - 1)).astype(int)  # Map steps to data_result indices

                        time_elapsed = 0
                        for i, idx in enumerate(indices):
                            if idx <= 1:
                                continue

                            time_start = time.time()

                            pct_completion = (i + 1) / total_frames

                            time_estimate = ""
                            if time_elapsed > 0 and pct_completion > 0.5:
                                est_seconds = int(time_elapsed * (1 - pct_completion) / pct_completion)
                                minutes = est_seconds // 60
                                seconds = est_seconds % 60
                                time_estimate = f"(Estimate: {str(minutes).zfill(2)}:{str(seconds).zfill(2)})"
                            
                            my_bar.progress(pct_completion, text=f"{progress_text} {100 * pct_completion:.1f}% {time_estimate}")
                            
                            # Get the Figure object
                            figure = dp.get_3_axis_plot(data_result, "mask", buffer_factor, do_topography=do_topography_mapping, restrain_topography_points=use_full_resolution, row_range=[0, idx])
                            
                            figure.update_layout(
                                paper_bgcolor="black",  # Black background outside the plot
                                plot_bgcolor="black",  # Black background inside the plot
                                font=dict(color="white")  # White text for visibility
                            )
                                                
                            buf = io.BytesIO()
                            figure.write_image(buf, format='png')
                            buf.seek(0)  # Reset buffer to start
                            frames.append(iio.imread(buf))
                            buf.close()

                            time_elapsed += time.time() - time_start
                    my_bar.empty()

                    gif_buffer = io.BytesIO()
                    loop_value = 0 if repeat_gif else 1  # 0 for infinite loop, 1 for no loop
                    iio.imwrite(gif_buffer, frames, format='GIF', fps=fps, loop=loop_value)
                    gif_buffer.seek(0)  # Reset buffer to start
                    st.image(gif_buffer)

                    st.download_button(
                        label="Download Gif",
                        data=gif_buffer,
                        file_name=f"{mask}_{timeline_start[0]}_{timeline_start[1]}.gif",
                        mime="gif",
                    )

    else:  # No lightning data
        st.warning("Data too restrained. Modify parameters on left sidebar.")

    st.divider()

    if st.button("Clear Cache"):
        st.write("Cache cleared")
        st.cache_data.clear()
        st.rerun()


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