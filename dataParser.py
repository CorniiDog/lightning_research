import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import numpy as np
import colorsys
import geopandas as gpd
import xarray, requests
from pyproj import Transformer
import math
import rioxarray
import time
import streamlit as st
import pickle as pkl

from typing import (
    Dict,
    Callable,
    Any,
    Tuple,
    List,
)  # For explicit types to rigid-ify the coding process

transformer_to_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978")
"""
Z-axis is aligned with the Earth's rotation axis, meaning it points through the North and South Poles.

X-axis points from the center of the Earth to the intersection of the Equator and the Prime Meridian (0° longitude).

Y-axis points from the center of the Earth to the intersection of the Equator and 90° East longitude.
"""

@st.cache_data
def get_start_date_label(lightning_data_folder: str, file_name: str, start_time_indicator = "Data start time:", start_time_format = "%m/%d/%y %H:%M:%S") -> str:
    """
    This retreieves the start date of a `.dat` file
    :param lightning_data_folder: The folder of the lightning data
    :param file_name: The name of the `.dat` file
    :param start_time_indicator: The text that indicates the start time of the data
    :param start_time_format: The formatting of the time upon reading the file, proceeding the text of `start_time_indicator`
    :return: String date and time is format of `"%m/%d@%H:%M"`
    """
    # Parse through data and retreive the Pandas DataFrame
    with open(os.path.join(lightning_data_folder, file_name), "r") as f:
        date_start: datetime = None

        for line in f:
            line: str  # Hint that line is a string
            if not date_start:
                if line.startswith(start_time_indicator):
                    potential_date = line.replace(start_time_indicator, "").strip() # Remove the indicator (i.e. "Data start time:")
                    date_time_obj = datetime.strptime(potential_date, start_time_format).replace(tzinfo=timezone.utc)

                    return date_time_obj.strftime("%m/%d %H:%M")
    return ""



######################################################################################################
## Helper functions below with processing and retreiving a nice-looking DataFrame
######################################################################################################
def parse_file(f, data_body_start = "*** data ***", start_time_indicator = "Data start time:", start_time_format = "%m/%d/%y %H:%M:%S", data_header_startswith = "Data:") -> pd.DataFrame:
    """
    This processes the entire file and extracts a pandas DataFrame

    :param f: The file object to read from
    :param data_body_start: The indicator for the start of the data body (that is when the information begins)
    :param start_time_format: The formatting of the time upon reading the file, proceeding the text of `start_time_indicator`
    :return pd.DataFrame: A pandas dataframe resembling the data
    """
    # Data from the file
    data_headers: list[str] = None
    data_result: pd.DataFrame = None
    date_start: datetime = None

    for line in f:  # Iterate through each line in the file
        line: str  # Hint that line is a string

        if not date_start:
            if line.startswith(start_time_indicator):
                potential_date = line.replace(start_time_indicator, "").strip() # Remove the indicator (i.e. "Data start time:")
                date_time_obj = datetime.strptime(potential_date, start_time_format).replace(tzinfo=timezone.utc)

                # Set the time to 00:00:00
                date_start = date_time_obj.replace(hour=0, minute=0, second=0, tzinfo=timezone.utc)

        # Extract data headers if not found
        if not data_headers:
            if line.startswith(data_header_startswith):
                data_headers = return_data_headers_if_found(
                    line=line, data_header_startswith=data_header_startswith
                )

        # If headers are found, then we go through
        elif data_result == None:
            if line.strip() == data_body_start:
                data_result = parse_data(f=f, data_headers=data_headers, date_start=date_start)

        # Assume fully indented the data and break the for loop
        else:
            break

    return data_result


seconds_since_start_of_day_header = "time (UT sec of day)"
"""
The indicator for the seconds since the start of dat (That is in universal time)

Default: `"time (UT sec of day)"`
"""


latitude_header = 'lat'
"""
The header for latitude

Default: `lat`
"""

process_handling: Dict[str, Callable[[str], Any]] = []
"""
Callback functions for processing based on header, for translation

Dict[str, Callable[[str], Any]] 

Basically means we are explicitly defining a dictionary named `process_handling`
with the key being a string `str`, and the value being a Callable object 
`Callable[[str], Any]` (basically a function). The callable function must have the 
parameter be a string but the return value can be anything.

Example:

```
process_handling: Dict[str, Callable[[str], Any]] = {
    "time (UT sec of day)": str_to_float,
    "lat": str_to_float,
    "lon": str_to_float,
    "alt(m)": str_to_float,
    "reduced chi^2": str_to_float,
    "P(dBW)": str_to_float,
    "mask": str_hex_to_int
}
```
"""

longitude_header = 'lon'
"""
The header for longitude

Default: `lon`
"""

altitude_meters_header = 'alt(m)'
"""
The header for altitude

Default: `alt(m)`
"""

def parse_data(
    f, data_headers: list[str], date_start: datetime) -> pd.DataFrame:
    """
    This goes through the remaining of a file and processes the entire data into something a lot easier to use in Python

    :param f: The file, at the immediate point of data begin
    :param data_headers: The headers to compile to
    :param date_start: The date of the start of data parsing
    :return pd.DataFrame: A pandas dataframe resembling the data
    """

    # Make dictionary (explicitly define that the key is a string and value is a list)
    dict_result: Dict[str, list] = {}

    dict_result["year"] = []
    dict_result["month"] = []
    dict_result["day"] = []
    dict_result["unix"] = []

    dict_result['x(m)'] = []
    dict_result['y(m)'] = []
    dict_result['z(m)'] = []

    # Make the keys for the dictionary with the designated headers
    for header in data_headers:
        dict_result[header] = []

    # Parse through the lines and apply
    for line in f:
        line: str  # Hint that line is a string

        data_row = line.split()  # Splits the line into designated sections
        respective_time = date_start

        lat, lon, alt = None, None, None

        # Process and format from string to respectable type
        for i in range(len(data_row)):
            data_cell = data_row[i]

            header = data_headers[i]

            # Format to respectable string
            if header in process_handling.keys():
                data_cell = process_handling[header](
                    data_cell
                )  # Process the data and parse it to designated format

            data_row[i] = data_cell

            if data_cell:
                if header == seconds_since_start_of_day_header:
                    respective_time += timedelta(seconds=data_cell)
                elif header == latitude_header:
                    lat = data_cell
                elif header == longitude_header:
                    lon = data_cell
                elif header == altitude_meters_header:
                    alt = data_cell


        # Append to dictionary
        for i in range(len(data_row)):
            dict_result[data_headers[i]].append(
                data_row[i]
            )  # Add data cell to the designated region

        x, y, z = 0.0, 0.0, 0.0
        if lat and lon and alt:
            x, y, z = transformer_to_ecef.transform(lat, lon, alt)
        

        dict_result["month"].append(respective_time.month)
        dict_result["day"].append(respective_time.day)
        dict_result["year"].append(respective_time.year)
        dict_result["unix"].append(respective_time.timestamp())
        dict_result["x(m)"].append(x)
        dict_result["y(m)"].append(y)
        dict_result["z(m)"].append(z)

    df = pd.DataFrame(dict_result) # Create dataframe
    
    return df

def return_data_headers_if_found(
    line: str, data_header_startswith: str
) -> list[str] | None:
    """
    This parses a line and determines if the designated location is headers

    :param line: This is a single string to parse from (i.e. `"Data: hello world"`)
    :param data_header_startswith: This is a single string to identify where to start (i.e. `"Data:"`)
    :return list[str] | None: A list of headers (`list[str]`) or `None`
    """

    data_header_startswith_len = len(data_header_startswith)

    if line.startswith(data_header_startswith):
        # Remove the start name ("Data:")
        line = line[data_header_startswith_len:]
        data_headers = line.split(",")  # Extract all data headers

        # Clean up and remove the extra spacing if it exists
        for i in range(len(data_headers)):
            data_headers[i] = data_headers[i].strip()

        return data_headers
    return None

lightning_data_cache_folder = "lightning_data_cache"

def polish_data_cache(lightning_data_folder: str):
    """
    The following polishes the data cache (removes .pkl files of corresponding lightning data un-present)
    """
    os.makedirs(lightning_data_folder, exist_ok=True)  # Ensure cache directory exists
    os.makedirs(lightning_data_cache_folder, exist_ok=True)  # Ensure cache directory exists


    with st.spinner("Polishing cache and updating data"):
        for cache_file in os.listdir(lightning_data_cache_folder):
            cache_filename_without_extension = cache_file.replace(".pkl", "")

            found = False
            for dat_file in os.listdir(lightning_data_folder):
                if cache_filename_without_extension in dat_file:
                    found = True
                    break

            if not found:
                os.remove(os.path.join(lightning_data_cache_folder, cache_file))

        
            
        

@st.cache_data
def get_dataframe_unfiltered(lightning_data_folder: str, file_name: str, data_body_start="*** data ***", start_time_indicator="Data start time:", start_time_format="%m/%d/%y %H:%M:%S", data_header_startswith="Data:", start_datetime: datetime = None, end_datetime: datetime = None) -> pd.DataFrame | None:
  os.makedirs(lightning_data_folder, exist_ok=True)  # Ensure cache directory exists
  os.makedirs(lightning_data_cache_folder, exist_ok=True)  # Ensure cache directory exists
  potential_cache = '.'.join(file_name.split(".")[:-1]) + ".pkl"
  potential_cache_full = os.path.join(lightning_data_cache_folder, potential_cache)

  data: pd.DataFrame = None

  start_datetime = start_datetime.replace(tzinfo=timezone.utc)
  end_datetime = end_datetime.replace(tzinfo=timezone.utc)

  if os.path.exists(potential_cache_full):
      with open(potential_cache_full, "rb") as f:
          # Load start_datetime, end_datetime, and data from the pickle file
          start_cached, end_cached, data = pkl.load(f)

          end_unix = end_datetime.timestamp()
          start_unix = start_datetime.timestamp()
          if not(start_unix <= start_cached <= end_unix or start_unix <= end_cached <= end_unix):
              return None
  else:
      with open(os.path.join(lightning_data_folder, file_name), "r") as f:
          data = parse_file(f=f, data_body_start=data_body_start, start_time_indicator=start_time_indicator, start_time_format=start_time_format, data_header_startswith=data_header_startswith)

          # Check if data is not empty
          if data is not None and not data.empty:
              start_cached = data['unix'].min()
              end_cached = data['unix'].max()
          else:
              start_cached = None
              end_cached = None

          with open(potential_cache_full, "wb") as f:
              # Dump [start_cached, end_cached, data] into the pickle file
              pkl.dump([start_cached, end_cached, data], f)
  return data
  
def get_dataframe(lightning_data_folder: str, file_name: str, count_required: Tuple[str, Callable], filters: Tuple[str, Callable], data_body_start="*** data ***", start_time_indicator="Data start time:", start_time_format="%m/%d/%y %H:%M:%S", data_header_startswith="Data:", start_datetime: datetime = None, end_datetime: datetime = None) -> pd.DataFrame | None:
    """
    Helper function for getting a pandas DataFrame from a .dat file.
    The function caches the data along with start and end datetime in a pickle file as [start_datetime, end_datetime, data].
    """
    os.makedirs(lightning_data_folder, exist_ok=True)  # Ensure cache directory exists

    data = get_dataframe_unfiltered(lightning_data_folder=lightning_data_folder, file_name=file_name, data_body_start=data_body_start, start_time_indicator=start_time_indicator, start_time_format=start_time_format, data_header_startswith=data_header_startswith, start_datetime=start_datetime, end_datetime=end_datetime)

    # Return None if data is empty
    if data is None:
        return None

    with st.spinner(f"Applying filters for `{file_name}`"):
        # Apply additional filters
        for column, condition in filters:
            data = data[data[column].apply(condition)]
    
    # Apply count_required filtering (Check if occurrences exceed condition)
    with st.spinner(f"Counting indices for `{file_name}`"):
        for column, condition in count_required:
            counts = data[column].value_counts()  # Get the count of occurrences for each value in the column
            valid_values = counts[counts.apply(condition)].index  # Get the values where count meets the condition
            data = data[data[column].isin(valid_values)]  # Filter the DataFrame by the valid values

    # If a time range is provided, filter the DataFrame based on the 'unix' time
    if start_datetime and end_datetime:
        # Ensure 'unix' column is in datetime format
        data = data[(data['unix'] >= start_datetime.timestamp()) & (data['unix'] <= end_datetime.timestamp())]

    return data

@st.cache_data
def generate_colors(num_colors):
    """Generate a list of unique dark colors using HSV/HSL color space."""
    colors = []
    for i in range(num_colors):
        hue = i / num_colors  # Hue is evenly spaced for unique colors
        saturation = 0.5  # Medium saturation for dark colors
        lightness = 0.3  # Dark lightness for dark colors
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append('#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
    return colors

def color_df(df: pd.DataFrame, identifier: str) -> pd.DataFrame:
    """Apply unique colors to rows in the DataFrame based on the `identifier` column values."""
    
    times = df[seconds_since_start_of_day_header]
    start_time = np.min(times)
    end_time = np.max(times)
    

    # Function to apply row color based on the identifier value
    def color_row(row):
        time_num = row[seconds_since_start_of_day_header]

        scalar = (time_num - start_time)/(end_time - start_time)

        color = '#{:02x}{:02x}{:02x}'.format(int(255), int(255-255*scalar), int(255-255*scalar))
        return [f'background-color: {color}' for _ in row]

    df.style.set_properties(**{'color': 'black'})

    # Apply the color function to entire rows and return the styled DataFrame
    return df.style.apply(color_row, axis=1)


def remove_empty_files_in_directory(directory):
    # Loop through all files in the given directory
    if not os.path.exists(directory):
        return
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if the item is a file (not a directory) and if its size is 0 bytes
        if os.path.isfile(file_path) and os.path.getsize(file_path) < 100:
            print(f"Removing empty file: {file_path}")
            os.remove(file_path)

demtypes = {
    "SRTMGL1": "SRTM GL1 30m",
    "AW3D30": "ALOS World 3D 30m",
    "NASADEM": "NASADEM Global DEM",
    "SRTMGL3": "SRTM GL3 90m",
    "COP30": "Copernicus Global DSM 30m",
    "USGS30m": "U.S. Geological Survey 3DEP raster dataset"
}


@st.cache_data
def get_opentopography_data(south, north, west, east, tif_file, demtype_index=0):
    """
    Fetch topography data from OpenTopography and load it as an xarray DataArray.
    
    :param south: Southern latitude of the bounding box
    :param north: Northern latitude of the bounding box
    :param west: Western longitude of the bounding box
    :param east: Eastern longitude of the bounding box
    :param demtype: Type of DEM data (default is SRTMGL3)
    :return: xarray DataArray containing the topography data
    """
    if demtype_index >= len(demtypes):
        return None

    # Get the API key from the environment variable
    api_key = os.getenv('OPENTOPOGRAPHY_API_KEY')

    if not api_key:
        raise ValueError("OPENTOPOGRAPHY_API_KEY environment variable is not set.")

    # Construct the request URL
    dem = list(demtypes.keys())[demtype_index]

    if dem == "USGS30m" or dem == "USGS10m":
        url = f"https://portal.opentopography.org/API/usgsdem?datasetName={dem}&south={south}&north={north}&west={west}&east={east}&outputFormat=GTiff&API_Key={api_key}"
    else:
        url = f"https://portal.opentopography.org/API/globaldem?demtype={dem}&south={south}&north={north}&west={west}&east={east}&outputFormat=GTiff&API_Key={api_key}"

    with st.spinner(text=f"Retreiving chunk data from: `{dem}: {demtypes[dem]}`"):
        # Send the GET request to OpenTopography API
        response = requests.get(url)

    # Check for successful response
    if response.status_code == 200:
        with open(tif_file, "wb") as f:
            f.write(response.content)
        
        if os.path.exists(tif_file):
            # Load the file as an xarray DataArray using rioxarray
            data_array = rioxarray.open_rasterio(filename=tif_file)
        
            return data_array

    else: # No data for the designated chunk or error
        with st.spinner(text=f"No topography data found for region or issue with `{dem}: {demtypes[dem]}`, trying different data"):
            time.sleep(2)
            topography_data = get_opentopography_data(south=south, north=north, west=west, east=east, tif_file=tif_file, demtype_index=demtype_index+1)
        return topography_data

@st.cache_data
def cache_topography_data(tif_file, params, tries = 0) -> xarray.DataArray | None:
    if os.path.exists(path=tif_file):
        print(f"Loading data from cache: {tif_file}")
        result = rioxarray.open_rasterio(filename=tif_file)
        
        if result is None:
            if tries > 5:
                return None
            os.remove(path=tif_file)
            return cache_topography_data(tif_file=tif_file, params=params, tries=tries+1)
        
        return result
    else:
        # Fetch the topography data if the pickle file does not exist
        with st.spinner("Fetching new data from OpenTopography..."):
            print("Fetching new data from OpenTopography...")
            # topo_data = Topography(**params)
            # topo_data.fetch()  # Download the data
            # da = topo_data.load()  # Load into xarray DataArray
            return get_opentopography_data(south=params['south'], north=params['north'], west=params['west'], east=params['east'], tif_file=tif_file)

cities_file: str | None = None
"""
The `.shp` file of the cities, within the folder of respective data downloaded.

For example: `cities_file = "ne_110m_populated_places/ne_110m_populated_places.shp"`

You can download data here: https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-populated-places/
"""

chunk_size = 4.0
"""
The chunk size to chunkify the topography data (i.e. `2.0` indicates chunks of lat/lon 2x2 chunks)
"""

def align_down(x: float, base: float = 1.0) -> int:
    """Aligns the number down to the nearest multiple of base and returns an integer."""
    return int(math.floor(x / base) * base)

def align_up(x: float, base: float = 1.0) -> int:
    """Aligns the number up to the nearest multiple of base and returns an integer."""
    return int(math.ceil(x / base) * base)

def generate_integer_chunks(params: Dict[str, float]) -> xarray.DataArray | None:
    
    # Align lat/lon boundaries to multiples of chunk_size
    lat_min = align_down(params["south"], base=chunk_size)
    lat_max = align_up(params["north"], base=chunk_size)
    lon_min = align_down(params["west"], base=chunk_size)
    lon_max = align_up(params["east"], base=chunk_size)

    # Loop through the latitude and longitude ranges using numpy's arange for floats
    for s in np.arange(lat_min, lat_max, chunk_size):
        for w in np.arange(lon_min, lon_max, chunk_size):
            # Ensure the chunk does not exceed the northern and eastern boundaries
            n = s + chunk_size
            e = w + chunk_size
            
            # Create a unique filename for caching
            tif_file = f"topography_cache/topography_{s}_{n}_{w}_{e}.tif"

            # Define the params for the current chunk
            p2 = params.copy()
            p2['north'] = float(n)
            p2['south'] = float(s)
            p2['east'] = float(e)
            p2['west'] = float(w)

            yield cache_topography_data(tif_file, p2)


@st.cache_data
def get_params(df:pd.DataFrame, buffer_factor = 0.0):
    lowest_lon = None
    highest_lon = None
    
    for longitude in df['lon']:
        if highest_lon == None or longitude > highest_lon:
            highest_lon = longitude
        if lowest_lon == None or longitude < lowest_lon:
            lowest_lon = longitude

    lowest_lat = None
    highest_lat = None
    for latitude in df['lat']:
        if highest_lat == None or latitude > highest_lat:
            highest_lat = latitude
        if lowest_lat == None or latitude < lowest_lat:
            lowest_lat = latitude

    params = {
        "north": highest_lat + buffer_factor,
        "south": lowest_lat - buffer_factor,
        "west": lowest_lon - buffer_factor,
        "east": highest_lon + buffer_factor,
    }

    return params

@st.cache_data
def add_topography(fig, df:pd.DataFrame, buffer_factor: float, lat=True, lon=True, alt=True):
    params = get_params(df=df, buffer_factor=buffer_factor)
    
    for da in generate_integer_chunks(params):

        # Convert the DataArray to a NumPy array for the topography surface plot
        elevation_data = da.values.squeeze()

        # Extract latitude and longitude from the DataArray
        latitudes = da.y.values
        longitudes = da.x.values

        with st.spinner("Restraining to a maximum of 1000 topography points"):
            forced_compression = 1000
            if len(longitudes) > forced_compression:
                secondary_downsampling = math.ceil(len(longitudes) / forced_compression)
                elevation_data_downsampled = elevation_data[::secondary_downsampling, ::secondary_downsampling]
                latitudes_downsampled = latitudes[::secondary_downsampling]
                longitudes_downsampled = longitudes[::secondary_downsampling]
            else:
                elevation_data_downsampled = elevation_data
                latitudes_downsampled = latitudes
                longitudes_downsampled = longitudes

        # Find indices of latitudes within the bounds
        lat_mask = (latitudes_downsampled >= params['south']) & (latitudes_downsampled <= params['north'])

        # Find indices of longitudes within the bounds
        lon_mask = (longitudes_downsampled >= params['west']) & (longitudes_downsampled <= params['east'])


        # Filter the latitude, longitude, and elevation arrays
        # Apply the latitude and longitude masks to create filtered arrays

        # Using np.ix_ to create a proper selection grid for 2D arrays
        filtered_elevation = elevation_data_downsampled[np.ix_(lat_mask, lon_mask)]
        filtered_latitudes = latitudes_downsampled[lat_mask]
        filtered_longitudes = longitudes_downsampled[lon_mask]

        if lat and lon and alt:
            fig.add_trace(go.Surface(
                z=filtered_elevation,
                x=filtered_longitudes,
                y=filtered_latitudes,
                colorscale='Viridis',
                opacity=0.9,
                showscale=True,
                
            ))
        elif lat and lon and not alt:
            fig.add_trace(go.Contour(
                z=filtered_elevation,
                x=filtered_longitudes,
                y=filtered_latitudes,
                colorscale='Viridis',
                showscale=True,
                opacity=0.9,
                contours=dict(
                    coloring='heatmap',
                    showlines=True,
                )
            ))
        
    return fig

interactive_2d_dot_size=5
"""
Presumably the 2d dot radius, in pixels(?)
"""

@st.cache_data
def get_interactive_2d_figure(df: pd.DataFrame, identifier: str, buffer_factor:float, do_topography=True, lat=True, lon=True, alt=False):
    """
    Create an interactive 2D plot based on the selected axes, colored by the identifier.

    :param df: DataFrame containing 'lat', 'lon', 'alt(m)', and identifier columns
    :param identifier: Column name to color code the data points
    :param do_topography: Boolean indicating whether to include topography data (only applicable for lat vs lon plot)
    :param lat: Boolean indicating whether to include latitude axis
    :param lon: Boolean indicating whether to include longitude axis
    :param alt: Boolean indicating whether to include altitude axis
    :return: Plotly figure object
    """
    # Determine the axes based on the parameters
    axes = []
    if lon:
        axes.append('lon')
    if alt:
        axes.append('alt(m)')
    if lat:
        axes.append('lat')

    if len(axes) != 2:
        raise ValueError("Exactly two of lat, lon, alt must be True for a 2D plot.")

    x_axis = axes[0]
    y_axis = axes[1]

    # Initialize figure
    fig = go.Figure()
    
    params = get_params(df, buffer_factor)

    lon_range = [params['west'], params['east']]
    lat_range = [params['south'], params['north']]

    if lat and lon:
        # If do_topography is True, plot the topography data
        if do_topography:
            fig = add_topography(fig, df,buffer_factor, lat, lon, alt)

        # If cities data is available, plot the cities
        if cities_file:
            gdf = gpd.read_file(cities_file)
            gdf = gdf.to_crs(epsg=4326)
            # Extract latitude and longitude from the geometry column
            gdf['latitude'] = gdf.geometry.y
            gdf['longitude'] = gdf.geometry.x
            # Filter the GeoDataFrame based on the bounding box and create a copy
            filtered_gdf = gdf[
                (gdf['latitude'] >= lat_range[0]) &
                (gdf['latitude'] <= lat_range[1]) &
                (gdf['longitude'] >= lon_range[0]) &
                (gdf['longitude'] <= lon_range[1])
            ].copy()

            if not filtered_gdf.empty:
                filtered_gdf.reset_index(drop=True, inplace=True)

                # Plot the cities
                fig.add_trace(go.Scatter(
                    x=filtered_gdf['longitude'],
                    y=filtered_gdf['latitude'],
                    mode='markers+text',
                    text=filtered_gdf['NAME'],
                    name='Cities',
                    textposition='top center',
                    marker=dict(size=interactive_2d_dot_size, color='red'),
                    showlegend=False
                ))
        fig.update_xaxes(range=lon_range, fixedrange=True)
        fig.update_yaxes(range=lat_range, fixedrange=True)
        fig.add_trace(go.Scatter(x=[lon_range[0], lon_range[0], lon_range[1], lon_range[1]], y=[lat_range[0], lat_range[1], lat_range[0], lat_range[1]], mode='markers',
                         marker=dict(opacity=0), showlegend=False))  # Invisible points

    elif lon and alt:
        fig.update_xaxes(range=lon_range, fixedrange=True)
        fig.add_trace(go.Scatter(x=lon_range, y=[0, 0], mode='markers',
                         marker=dict(opacity=0), showlegend=False))  # Invisible points
    elif lat and alt:
        fig.update_yaxes(range=lat_range, fixedrange=True)    # Generate colors for the data points
        fig.add_trace(go.Scatter(x=[0, 0], y=lat_range, mode='markers',
                         marker=dict(opacity=0), showlegend=False))  # Invisible points
        
    # Limit the number of points for performance
    max_points = 10000  # Adjust based on performance needs
    if len(df) > max_points:
        df_sampled = df.sample(n=max_points, random_state=42)
        st.warning(f"Sampling {max_points} out of {len(df)} data points for plotting.")
    else:
        df_sampled = df

    # Plot the data points
    fig.add_trace(go.Scatter(
        x=df_sampled[x_axis],
        y=df_sampled[y_axis],
        mode='markers',
        marker=dict(
                size=interactive_2d_dot_size,
                color=df_sampled[seconds_since_start_of_day_header],
                colorscale=[
                    [0.0, 'rgb(255, 0, 0)'],
                    [1.0, 'rgb(255, 255, 255)']
                ],
                opacity=1
            ),
        name=f'{identifier} {df_sampled[identifier][0]}',
        showlegend=False
    ))
    

    # Update the layout
    fig.update_layout(
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        height=600,
        legend_title=identifier
    )

    return fig

@st.cache_data
def get_3_axis_plot(df:pd.DataFrame, identifier:str, buffer_factor:float, do_topography=True):
    lonalt_fig = get_interactive_2d_figure(df, identifier, buffer_factor, do_topography=do_topography, lat=False, lon=True, alt=True)
    latlon_fig = get_interactive_2d_figure(df, identifier, buffer_factor, do_topography=do_topography, lat=True, lon=True, alt=False)
    latalt_fig = get_interactive_2d_figure(df, identifier, buffer_factor, do_topography=do_topography, lat=True, lon=False, alt=True)

    fig_combined = make_subplots(
        rows=3, cols=3,
        specs=[[{"type": "scatter", "colspan": 2}, None, None], [{"type": "scatter", "colspan": 2, "rowspan": 2}, None, {"type": "scatter", "rowspan": 2}], [None, None, None]],  # Empty cell in the top-right
        vertical_spacing=0.1, horizontal_spacing=0.1
    )

    # Add the traces from fig1 (Latitude vs Longitude) into the combined plot
    for trace in latlon_fig.data:
        fig_combined.add_trace(trace, row=2, col=1)

    # Add the traces from fig2 (Longitude vs Altitude) into the combined plot
    for trace in lonalt_fig.data:
        fig_combined.add_trace(trace, row=1, col=1)

    # Add the traces from fig3 (Altitude vs Latitude) into the combined plot
    for trace in latalt_fig.data:
        fig_combined.add_trace(trace, row=2, col=3)

    # Update axis titles
    fig_combined.update_xaxes(title_text="Longitude", row=2, col=1)
    fig_combined.update_yaxes(title_text="Latitude", row=2, col=1)

    fig_combined.update_xaxes(title_text="Longitude", row=1, col=1)
    fig_combined.update_yaxes(title_text="Altitude (m)", row=1, col=1)

    fig_combined.update_xaxes(title_text="Altitude (m)", row=2, col=3)
    fig_combined.update_yaxes(title_text="Latitude", row=2, col=3)

    # Update the layout
    fig_combined.update_layout(height=800, width=800, title_text="Combined Plot of Lightning Strikes")

    return fig_combined

interactive_3d_dot_size=5
"""
Presumably the 3d dot radius, in pixels(?)
"""

@st.cache_data
def get_interactive_3d_figure(df: pd.DataFrame, identifier: str, buffer_factor: float, do_topography=True):
    """
    Create an interactive 3D scatter plot for latitude, longitude, and altitude, colored by mask.

    :param df: DataFrame containing lat, long, alt, and mask data
    :param identifier: Column to color the plot based on
    :return: Plotly figure object
    """

    fig = go.Figure()

    os.makedirs("topography_cache", exist_ok=True)  # Ensure cache directory exists

    params = get_params(df, buffer_factor)

    if do_topography:
        fig = add_topography(fig, df, buffer_factor)


    # If cities are able to be loaded, then load cities
    if cities_file:
        gdf = gpd.read_file(cities_file)
        gdf = gdf.to_crs(epsg=4326)

        # Extract latitude and longitude from the geometry column
        gdf['latitude'] = gdf.geometry.y
        gdf['longitude'] = gdf.geometry.x

        # Filter the GeoDataFrame based on the bounding box and create a copy
        filtered_gdf = gdf[
            (gdf['latitude'] >= params['south']) &
            (gdf['latitude'] <= params['north']) &
            (gdf['longitude'] >= params['west']) &
            (gdf['longitude'] <= params['east'])
        ].copy()  # Added .copy() here

        if not filtered_gdf.empty:
            filtered_gdf.reset_index(drop=True, inplace=True)

            # Safely assign a new column using .loc to avoid the warning
            filtered_gdf.loc[:, 'altitude'] = 0

            # Add cities as Scatter3d
            fig.add_trace(go.Scatter3d(
                x=filtered_gdf['longitude'],
                y=filtered_gdf['latitude'],
                z=filtered_gdf['altitude'],
                mode='text',
                name='Populated Places',
                hoverinfo='text',
                hovertext=filtered_gdf['NAME'],
                text = filtered_gdf['NAME'],
                showlegend=False,
                
            ))

    #  Limit the number of scatter points (lightning data) for faster plotting
    max_points = 1000  # Adjust based on performance needs
    if len(df) > max_points:
        df_sampled = df.sample(n=max_points, random_state=42)
        print(f"Sampling {max_points} out of {len(df)} lightning points for plotting.")
    else:
        df_sampled = df

    #  Overlay the lightning data (scatter points) on top of the topography surface
    fig.add_trace(go.Scatter3d(
            x=df_sampled['lon'],    # Longitude corresponds to x-axis
            y=df_sampled['lat'],    # Latitude corresponds to y-axis
            z=df_sampled['alt(m)'], # Altitude corresponds to z-axis
            mode='markers',
            marker=dict(
                size=interactive_3d_dot_size,
                color=df_sampled[seconds_since_start_of_day_header],
                colorscale=[
                    [0.0, 'rgb(255, 0, 0)'],
                    [1.0, 'rgb(255, 255, 255)']
                ],
                opacity=1
            ),
            name=f'{identifier} {df_sampled["mask"][0]}',
            showlegend=False
        ))


    fig.update_layout(
    scene=dict(
        xaxis=dict(
            title='Longitude',
            gridcolor='lightgray',  # Lighter color for gridlines
            zerolinecolor='lightgray',  # Lighter color for axis lines
            range=[params['west'], params['east']]
        ),
        yaxis=dict(
            title='Latitude',
            gridcolor='lightgray',  # Lighter color for gridlines
            zerolinecolor='lightgray',  # Lighter color for axis lines
            range=[params['south'], params['north']]
        ),
        zaxis=dict(
            title='Altitude (m)',
            gridcolor='lightgray',  # Lighter color for gridlines
            zerolinecolor='lightgray',  # Lighter color for axis lines
            nticks=10,
        )
    ),
    height=400,
    )

    return fig

speed_of_light: float = 299792458.0 # m/s
"""
The literal representing the speed of light in meters per second (m/s). May
be changed to be more precise.

Default: `299792458.0`
"""

@st.cache_data
def get_strikes(df: pd.DataFrame, lightning_max_strike_time: float = 0.15, lightning_max_strike_distance: float = 3000.0, lightning_minimum_speed: float = 299792.458, min_points_for_lightning = 2) -> Tuple[List[pd.DataFrame], List[Tuple[str, str]]]:
    """
    Gets the list of strikes.

    :param df: The strike data dataframe
    :param lightning_max_strike_time: Max strike time, in seconds
    :param lightning_max_strike_distance: The max strike distance, in meters
    :param lightning_minimum_speed: The minimum speed to be monitored and accepted in meters per second (m/s). Recommended to be around 1-2% of the speed of light as lightning is traditionally slower than the speed of light due to environmental factors.
    :param min_points_for_lightning: The minimum number of points required for a lightning strike identification
    """

    # We create a dictionary so that we can do hashing using the mask as the key for the hash. 
    # This improves processing speed by around 5-7 times, depending on the number of
    # elements sharing the same mask
    mask_locations: Dict[str, List[pd.DataFrame]] = {}
    mask_strike_times: Dict[str, List[Tuple[float, float]]]= {}

    for i, row in df.iterrows():
        x1 = row['x(m)']
        y1 = row['y(m)']
        z1 = row['z(m)']
        time1 = row['unix']
        mask_value = row['mask']
            
        data_found = False

        if mask_value not in mask_locations.keys():
            mask_locations[mask_value] = []
            mask_strike_times[mask_value] = []
        

        for j in range(len(mask_locations[mask_value])):
            if data_found:
                break
            
            # Check if the new point's time is within acceptable range of the existing strike's time boundaries
            min_time, max_time = mask_strike_times[mask_value][j]
            if time1 < min_time - lightning_max_strike_time or time1 > max_time + lightning_max_strike_time:
                continue  # Skip this strike as the times are too different

            # Iterate over rows in the existing strike DataFrame
            for _, other_row in mask_locations[mask_value][j].iterrows():
                time2 = other_row['unix']
                delta_t = time1 - time2

                # If times are grossly different, skip
                if np.abs(delta_t) > lightning_max_strike_time:
                    continue

                x2 = other_row['x(m)']
                y2 = other_row['y(m)']
                z2 = other_row['z(m)']
                
                rough_dist = max(abs(x1 - x2), abs(y1 - y2), abs(z1 - z2))
                if rough_dist > lightning_max_strike_distance:
                    continue

                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

                # If distance out of bounds, skip
                if dist > lightning_max_strike_distance:
                    continue

                speed = np.abs(dist / delta_t) if delta_t != 0 else np.inf

                # If speed is unrealistic, skip
                if speed < lightning_minimum_speed or speed > speed_of_light:
                    continue

                # Concatenate the row to the existing DataFrame
                mask_locations[mask_value][j] = pd.concat([mask_locations[mask_value][j], pd.DataFrame([row])], ignore_index=True)

                # Update the time boundaries for this strike
                mask_strike_times[mask_value][j] = (min(min_time, time1), max(max_time, time1))

                data_found = True
                break

        # If the row didn't match any existing strike, start a new one
        if not data_found:
            mask_locations[mask_value].append(pd.DataFrame([row]))
            mask_strike_times[mask_value].append((time1, time1))  # Initialize min and max time with time1
    

    lightning_strikes: List[pd.DataFrame] = []
    strike_times: List[Tuple[float, float]] = []  # List of (min_time, max_time) for each strike
    for mask_value, strikes in mask_locations.items():
        lightning_strikes += strikes
        strike_times += mask_strike_times[mask_value]


    with st.spinner("Filtering strikes"):
      # Filter out strikes with fewer rows than min_points_for_lightning
      filtered_strikes = []
      filtered_strike_times = []
      for i, strike in enumerate(lightning_strikes):
          if len(strike) >= min_points_for_lightning:
              filtered_strikes.append(strike)
              # Convert Unix timestamps to ISO formatted datetime strings
              min_time_unix, max_time_unix = strike_times[i]
              min_time_str = datetime.fromtimestamp(min_time_unix, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
              max_time_str = datetime.fromtimestamp(max_time_unix, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
              filtered_strike_times.append((min_time_str, max_time_str))

    return filtered_strikes, filtered_strike_times
