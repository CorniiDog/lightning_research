import pandas as pd
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import numpy as np
import colorsys, pickle
from bmi_topography import Topography
import geopandas as gpd
import rasterio
from pyproj import Transformer
import math
import streamlit as st

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


params = Topography.DEFAULT.copy()
params["south"] = 25.84  # Modify these coordinates for your area of interest (Texas)
params["north"] = 36.50
params["west"] = -106.65
params["east"] = -93.51
params['dem_type'] = 'SRTMGL3'
params['cache_dir'] = '~/.bmi_topography'

# Essential functions to use
count_required = []
"""
This determines count instances rule.

That is, if you provide a header, you provide a function for the count of items.

For example, for `1 >= n >= 2` that requires instances to be between `1` and `2` occurances.

Example:

```
count_required = [
    ["mask", mask_count_rule]
]
```
"""

process_handling: Dict[str, Callable[[str], Any]] = {
    "time (UT sec of day)": lambda my_str: float(my_str),  # Convert to float
    "lat": lambda my_str: float(my_str),  # Convert to float
    "lon": lambda my_str: float(my_str),  # Convert to float
    "alt(m)": lambda my_str: float(my_str),  # Convert to float
    "reduced chi^2": lambda my_str: float(my_str),  # Convert to float
    "P(dBW)": lambda my_str: float(my_str), # Convert to float
    "mask": lambda hex_str: int(hex_str, 16) # Convert the hex-code mask to decimal
}
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

# i.e. Data: time (UT sec of day), lat, lon, alt(m), reduced chi^2, P(dBW), mask
data_header_startswith = "Data:"

# After conversion process, you can now add callback functions for filters
# Accepts the designated row if the function returns true
#
# List[str, Callable]
# We are explicitly defining a list (or array) to the variable `filters`
filters: List = []
# Example:
"""
After conversion process, you can now add callback functions for filters
Accepts the designated row if the function returns true

List[str, Callable]
We are explicitly defining a list (or array) to the variable `filters`
Example:

```
filters: List = [
    ["reduced chi^2", accept_chi_below],
    ["alt(m)", data_above_20_km]
]
```
"""

# i.e. Data start time: 09/11/24 20:40:00
start_time_indicator = "Data start time:"
"""
The text that indicates the start time of the data
"""

start_time_format = "%m/%d/%y %H:%M:%S"
"""
The formatting of the time upon reading the file, proceeding the text of `start_time_indicator`

Default: `"%m/%d/%y %H:%M:%S"`
"""

# i.e. *** data ***
data_body_start = "*** data ***"
"""
The indicator for the start of the data body (that is when the information begins)

Default: `"*** data ***"`
"""

######################################################################################################
## Helper functions below with processing and retreiving a nice-looking DataFrame
######################################################################################################
def parse_file(f) -> pd.DataFrame:
    """
    This processes the entire file and extracts a pandas DataFrame

    :param f: The file object to read from
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
                date_time_obj = datetime.strptime(potential_date, start_time_format)

                # Set the time to 00:00:00
                date_start = date_time_obj.replace(hour=0, minute=0, second=0)

        # Extract data headers if not found
        if not data_headers:
            if line.startswith(data_header_startswith):
                data_headers = return_data_headers_if_found(
                    line, data_header_startswith
                )

        # If headers are found, then we go through
        elif data_result == None:
            if line.strip() == data_body_start:
                data_result = parse_data(f, data_headers, date_start)

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

start_datetime: datetime = None
"""
The start date window, as a datetime, for calculations
"""

end_datetime: datetime = None
"""
The end date window, as a datetime, for calculations
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

    # Create counter object with initialization of data
    counters = {}
    for arr in count_required:
        arr: List
        header = arr[0]
        counters[header] = {} # Add the header to the counter

    # Make the keys for the dictionary with the designated headers
    for header in data_headers:
        dict_result[header] = []

    # Parse through the lines and apply
    for line in f:
        line: str  # Hint that line is a string

        data_row = line.split()  # Splits the line into designated sections
        respective_time = date_start
        
        # If start_date is established and that the respective time is before the start date, then skip
        if start_datetime and respective_time < start_datetime:
            continue
        
        # If end_date is established and that the respective time is after the end date, then skip
        if end_datetime and respective_time > end_datetime:
            continue

        lat, lon, alt = None, None, None

        # Process and format from string to respectable type
        allow_pass = True
        for i in range(len(data_row)):
            data_cell = data_row[i]

            header = data_headers[i]

            # Format to respectable string
            if header in process_handling.keys():
                data_cell = process_handling[header](
                    data_cell
                )  # Process the data and parse it to designated format

                # Add seconds if the cell is respective

            # Increment counter for the designated header (this is for counter filtering)
            if header in counters.keys():
                if not data_cell in counters[header].keys():
                    counters[header][data_cell] = 0
                
                counters[header][data_cell] += 1

            # Ensure that it's within filters rules. If it's not then prevent padding and break
            for j in range(len(filters)):
                if filters[j][0] == header and not filters[j][1](data_cell): # If the header name is the same (0 index)
                    allow_pass = False #Use as flag for allow_pass
                    break

            # Break if allow_pass flag is false
            if not allow_pass:
                break

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


        # Skip the line and do not allow parsing if the allow_pass is false
        if not allow_pass:
            continue

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

    # Go through every dataframe row
    for index, row in df.iterrows():
        # For every counter category headers
        for header, data in counters.items():
            
            num_instances = data[row[header]]

            for count_arr in count_required:
                header: str = count_arr[0]
                callback_func: Callable = count_arr[1]
                # Now compare and drop if it fails
                if not callback_func(num_instances):
                    df.drop(index, inplace=True)
    
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


def get_dataframe(lightning_data_folder: str, file_name: str) -> pd.DataFrame | None:
    """
    Helper function for getting a pandas DataFrame from a .dat file
    """

    # Parse through data and retreive the Pandas DataFrame
    with open(os.path.join(lightning_data_folder, file_name), "r") as f:
        return parse_file(f)
    return None


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
    # Get the unique mask/identifier values
    unique_values = df[identifier].unique()

    # Assign a unique dark color to each unique value
    value_colors = {val: color for val, color in zip(unique_values, generate_colors(len(unique_values)))}

    # Function to apply row color based on the identifier value
    def color_row(row):
        value = row[identifier]
        if value in value_colors:
            return [f'background-color: {value_colors[value]}' for _ in row]
        return [''] * len(row)

    # Apply the color function to entire rows and return the styled DataFrame
    return df.style.apply(color_row, axis=1)

def cache_topography_data(pickle_file, params) -> pd.DataFrame:
    # Check if the pickle file already exists
    if os.path.exists(pickle_file):
        print(f"Loading data from cache: {pickle_file}")
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    else:
        # Fetch the topography data if the pickle file does not exist
        with st.spinner("Fetching new topography data from OpenTopography..."):
            print("Fetching new topography data from OpenTopography...")
            topo_data = Topography(**params)
            topo_data.fetch()  # Download the data
            da = topo_data.load()  # Load into xarray DataArray

        # Save the data to a pickle file for future use
        with open(pickle_file, 'wb') as f:
            pickle.dump(da, f)
        print(f"Data cached as: {pickle_file}")
        return da

def get_tile_indices(south: float, north: float, west: float, east: float, tile_size: float = 1.0) -> List[Tuple[int, int]]:
    """
    Calculate the list of tile indices that intersect with the given bounding box.

    :param south: Southern latitude boundary
    :param north: Northern latitude boundary
    :param west: Western longitude boundary
    :param east: Eastern longitude boundary
    :param tile_size: Size of each tile in degrees (default is 1°)
    :return: List of tuples representing tile indices (min_lat, min_lon)
    """
    tiles = []
    lat_start = int(np.floor(south / tile_size) * tile_size)
    lat_end = int(np.floor(north / tile_size) * tile_size)
    lon_start = int(np.floor(west / tile_size) * tile_size)
    lon_end = int(np.floor(east / tile_size) * tile_size)

    for lat in range(lat_start, lat_end + 1, int(tile_size)):
        for lon in range(lon_start, lon_end + 1, int(tile_size)):
            tiles.append((lat, lon))
    return tiles

def cache_topography_tile(pickle_dir: str, tile: Tuple[int, int], params: Dict[str, Any]) -> rasterio.DatasetReader:
    """
    Cache a single topography tile. Download and save it if not already cached.

    :param pickle_dir: Directory where pickle files are stored
    :param tile: Tuple representing the tile indices (min_lat, min_lon)
    :param params: Parameters for fetching topography data
    :return: Rasterio DatasetReader object for the tile
    """
    min_lat, min_lon = tile
    tile_key = f"tile_{min_lat}_{min_lon}.pkl"
    tile_path = os.path.join(pickle_dir, tile_key)

    if os.path.exists(tile_path):
        print(f"Loading tile from cache: {tile_key}")
        with open(tile_path, 'rb') as f:
            tile_data: rasterio.DatasetReader = pickle.load(f)
    else:
        print(f"Fetching new topography data for tile: {tile_key}")
        # Define the tile-specific bounding box
        tile_params = params.copy()
        tile_params["south"] = min_lat
        tile_params["north"] = min_lat + 1
        tile_params["west"] = min_lon
        tile_params["east"] = min_lon + 1

        # Initialize and fetch topography data for the tile
        topo_data = Topography(**tile_params)
        topo_data.fetch()  # Download the data
        tile_data = topo_data.load()  # Load into xarray DataArray

        # Save the tile data to cache
        with open(tile_path, 'wb') as f:
            pickle.dump(tile_data, f)
        print(f"Tile cached as: {tile_path}")

    return tile_data


cities_file: str | None = None
"""
The `.shp` file of the cities, within the folder of respective data downloaded.

For example: `cities_file = "ne_110m_populated_places/ne_110m_populated_places.shp"`

You can download data here: https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-populated-places/
"""

chunk_size = 2.0
"""
The chunk size to chunkify the topography data (i.e. `2.0` indicates chunks of lat/lon 2x2 chunks)
"""

def align_down(x: float, base: float = 1.0) -> int:
    """Aligns the number down to the nearest multiple of base and returns an integer."""
    return int(math.floor(x / base) * base)

def align_up(x: float, base: float = 1.0) -> int:
    """Aligns the number up to the nearest multiple of base and returns an integer."""
    return int(math.ceil(x / base) * base)

def generate_integer_chunks(
    south: float, north: float, west: float, east: float, chunk_size: float = 2.0
) -> Dict[str, int]:
    """
    Generates non-overlapping chunks aligned to integer boundaries.

    :param south: Southern latitude boundary
    :param north: Northern latitude boundary
    :param west: Western longitude boundary
    :param east: Eastern longitude boundary
    :param chunk_size: Size of each chunk in degrees (default is 2.0)
    :yield: Dictionary containing chunk boundaries as integers
    """
    # Align overall boundaries to integer degrees
    aligned_south = align_down(south, base=1.0)
    aligned_north = align_up(north, base=1.0)
    aligned_west = align_down(west, base=1.0)
    aligned_east = align_up(east, base=1.0)

    # Iterate over latitude in steps of chunk_size
    lat_start = aligned_south
    while lat_start < aligned_north:
        chunk_south = lat_start
        chunk_north = lat_start + chunk_size
        # Ensure the northern boundary does not exceed the original limit
        if chunk_north > north:
            chunk_north = align_up(north, base=1.0)
        else:
            chunk_north = int(chunk_north)  # Ensure integer

        # Iterate over longitude in steps of chunk_size
        lon_start = aligned_west
        while lon_start < aligned_east:
            chunk_west = lon_start
            chunk_east = lon_start + chunk_size
            # Ensure the eastern boundary does not exceed the original limit
            if chunk_east > east:
                chunk_east = align_up(east, base=1.0)
            else:
                chunk_east = int(chunk_east)  # Ensure integer

            yield {
                'south': chunk_south,
                'north': chunk_north,
                'west': chunk_west,
                'east': chunk_east
            }

            lon_start += chunk_size
        lat_start += chunk_size

downsampling_factor = 10
"""
The downsampling factor for the topography data (`10` implies we only accept 1 every 10 datapoints into the graph)

It's meant to act as an optomization technique
"""

def add_topography(fig, df:pd.DataFrame, lat=True, lon=True, alt=True):
    lowest_lon = None
    largest_lon = None
    for longitude in df['lon']:
        if largest_lon == None or longitude > largest_lon:
            largest_lon = longitude
        if lowest_lon == None or longitude < lowest_lon:
            lowest_lon = longitude
    
    params["west"] = lowest_lon - 1
    params["east"] = largest_lon + 1

    lowest_lat = None
    highest_lat = None
    for latitude in df['lat']:
        if highest_lat == None or latitude > highest_lat:
            highest_lat = latitude
        if lowest_lat == None or latitude < lowest_lat:
            lowest_lat = latitude

    params["south"] = lowest_lat - 1
    params["north"] = highest_lat + 1
    
    for idx, chunk in enumerate(generate_integer_chunks(params['south'], params['north'], params['west'], params['east'], chunk_size), start=1):
        bbox_key = f"{chunk['south']}_{chunk['north']}_{chunk['west']}_{chunk['east']}"
        pickle_file = f"topography_cache/{bbox_key}.pkl"

        # Step 1: Cache or load the topography data
        da = cache_topography_data(pickle_file, params)

        # Step 2: Convert the DataArray to a NumPy array for the topography surface plot
        elevation_data = da.values.squeeze()

        # Extract latitude and longitude from the DataArray
        latitudes = da.y.values
        longitudes = da.x.values

        # Step 3: Downsample the topography data for faster plotting
        # Adjust the factor based on your performance needs
        elevation_data_downsampled = elevation_data[::downsampling_factor, ::downsampling_factor]
        latitudes_downsampled = latitudes[::downsampling_factor]
        longitudes_downsampled = longitudes[::downsampling_factor]

        if lat and lon and alt:
            fig.add_trace(go.Surface(
                z=elevation_data_downsampled,
                x=longitudes_downsampled,
                y=latitudes_downsampled,
                colorscale='Viridis',
                opacity=0.7,
                showscale=True
            ))
        elif lat and lon and not alt:
            fig.add_trace(go.Contour(
                z=elevation_data_downsampled,
                x=longitudes_downsampled,
                y=latitudes_downsampled,
                colorscale='Viridis',
                showscale=True,
                opacity=0.7,
                contours=dict(
                    coloring='heatmap',
                    showlines=False,
                )
            ))
        
    return fig

def get_interactive_2d_figure(df: pd.DataFrame, identifier: str, do_topography=True, lat=True, lon=True, alt=False):
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

    if {'lon', 'lat'} == set(axes):
        # If do_topography is True, plot the topography data
        if do_topography:
            fig = add_topography(fig, df, lat, lon, alt)

        # If cities data is available, plot the cities
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
                    marker=dict(size=8, color='red'),
                    showlegend=False
                ))
        fig.update_layout(
        xaxis=dict(range=[min(params['west'], np.min(df['lon'])), max(params['east'], np.max(df['lon']))]),
        yaxis=dict(range=[min(params['south'], np.min(df['lat'])), max(params['north'], np.max(df['lon']))])
        )

    elif {'lon', 'alt(m)'} == set(axes):
        fig.update_layout(
            xaxis=dict(range=[min(params['west'], np.min(df['lon'])), max(params['east'], np.max(df['lon']))])
        )
    elif {'alt(m)', 'lat'} == set(axes):
        fig.update_layout(
            yaxis=dict(range=[min(params['south'], np.min(df['lat'])), max(params['north'], np.max(df['lon']))])
        )

    # Generate colors for the data points
    unique_values = df[identifier].unique()
    value_colors = {val: color for val, color in zip(unique_values, generate_colors(len(unique_values)))}

    # Limit the number of points for performance
    max_points = 10000  # Adjust based on performance needs
    if len(df) > max_points:
        df_sampled = df.sample(n=max_points, random_state=42)
        print(f"Sampling {max_points} out of {len(df)} data points for plotting.")
    else:
        df_sampled = df

    # Plot the data points
    for value, color in value_colors.items():
        subset = df_sampled[df_sampled[identifier] == value]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset[x_axis],
            y=subset[y_axis],
            mode='markers',
            marker=dict(size=8, color=color, opacity=0.7),
            name=f'{identifier} {value}',
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
    

def get_interactive_3d_figure(df: pd.DataFrame, identifier: str, do_topography=True):
    """
    Create an interactive 3D scatter plot for latitude, longitude, and altitude, colored by mask.

    :param df: DataFrame containing lat, long, alt, and mask data
    :param identifier: Column to color the plot based on
    :return: Plotly figure object
    """

    # Topography generation
    # Topography generation
    # Generate a unique filename based on the bounding box

    fig = go.Figure()

    os.makedirs("topography_cache", exist_ok=True)  # Ensure cache directory exists


    if do_topography:
        fig = add_topography(fig, df)


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
                showlegend=False

            ))



    # Step 5: Generate colors for scatter points (lightning data)
    unique_values = df[identifier].unique()
    value_colors = {val: color for val, color in zip(unique_values, generate_colors(len(unique_values)))}

    # Step 6: Limit the number of scatter points (lightning data) for faster plotting
    max_points = 1000  # Adjust based on performance needs
    if len(df) > max_points:
        df_sampled = df.sample(n=max_points, random_state=42)
        print(f"Sampling {max_points} out of {len(df)} lightning points for plotting.")
    else:
        df_sampled = df

    # Step 7: Overlay the lightning data (scatter points) on top of the topography surface
    for mask_value, color in value_colors.items():
        subset = df_sampled[df_sampled[identifier] == mask_value]  # Subset of data matching the current mask_value
        if subset.empty:
            continue  # Skip if no data for this mask_value
        fig.add_trace(go.Scatter3d(
            x=subset['lon'],    # Longitude corresponds to x-axis
            y=subset['lat'],    # Latitude corresponds to y-axis
            z=subset['alt(m)'], # Altitude corresponds to z-axis
            mode='markers',
            marker=dict(
                size=5,
                color=color,
                opacity=0.9
            ),
            name=f'{identifier} {mask_value}',
            showlegend=False
        ))

        # Draw lines between the masks
        # fig.add_trace(go.Scatter3d(
        #     x=subset['lon'],
        #     y=subset['lat'],
        #     z=subset['alt(m)'],
        #     mode='lines',
        #     line=dict(color=color, width=2),
        #     name=f'{identifier} {mask_value} lines',
        #     showlegend=False
        # ))

    fig.update_layout(
    scene=dict(
        xaxis=dict(
            title='Longitude',
            gridcolor='lightgray',  # Lighter color for gridlines
            zerolinecolor='lightgray',  # Lighter color for axis lines
            range=[min(params['west'], np.min(df['lon'])), max(params['east'], np.max(df['lon']))]
        ),
        yaxis=dict(
            title='Latitude',
            gridcolor='lightgray',  # Lighter color for gridlines
            zerolinecolor='lightgray',  # Lighter color for axis lines
            range=[min(params['south'], np.min(df['lat'])), max(params['north'], np.max(df['lon']))]
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

lightning_max_strike_time: float = 0.15
"""
Max strike time, in seconds

Default: `0.15`
"""

lightning_max_strike_distance: float = 3000.0
"""
The max strike distance, in meters

Default: `3000.0`
"""

lightning_minimum_speed: float = 299792.458
"""
The minimum speed to be monitored and accepted in meters per second (m/s). Recommended
to be around 1-2% of the speed of light as lightning is
traditionally slower than the speed of light due to
environmental factors.

Default: `299792.458` (The speed of light is)
"""

speed_of_light: float = 299792458.0 # m/s
"""
The literal representing the speed of light in meters per second (m/s). May
be changed to be more precise.

Default: `299792458.0`
"""

min_points_for_lightning = 2
"""
The minimum number of points required for a lightning strike identification
"""

@st.cache_data
def get_strikes(df: pd.DataFrame) -> Tuple[List[pd.DataFrame], List[Tuple[str, str]]]:
    """
    
    """
    lightning_strikes: List[pd.DataFrame] = []
    strike_times: List[Tuple[float, float]] = []  # List of (min_time, max_time) for each strike

    unique_masks = df['mask'].unique()

    # Iterate through each unique mask value
    for mask_value in unique_masks:
        # Filter the DataFrame to get only rows with the current mask value
        mask_subset = df[df['mask'] == mask_value]
        
        for i in range(len(mask_subset)):
            # Use double brackets to get a DataFrame instead of a Series
            row = mask_subset.iloc[[i]]  

            x1 = row['x(m)'].values[0]
            y1 = row['y(m)'].values[0]
            z1 = row['z(m)'].values[0]
            time1 = row['unix'].values[0]

            data_found = False  # Flag to check if row is added to an existing strike

            for j in range(len(lightning_strikes)):
                # Check if the new point's time is within acceptable range of the existing strike's time boundaries
                min_time, max_time = strike_times[j]
                if time1 < min_time - lightning_max_strike_time or time1 > max_time + lightning_max_strike_time:
                    continue  # Skip this strike as the times are too different

                if data_found:
                    break

                # Iterate over rows in the existing strike DataFrame
                for k, other_row in lightning_strikes[j].iterrows():
                    if mask_value != other_row['mask']:
                        break

                    time2 = other_row['unix']
                    delta_t = time1 - time2

                    # If times are grossly different, skip
                    if np.abs(delta_t) > lightning_max_strike_time:
                        continue


                    x2 = other_row['x(m)']
                    y2 = other_row['y(m)']
                    z2 = other_row['z(m)']
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

                    # If distance out of bounds, skip
                    if dist > lightning_max_strike_distance:
                        continue

                    speed = np.abs(dist / delta_t) if delta_t != 0 else np.inf

                    # If speed is unrealistic, skip
                    if speed < lightning_minimum_speed or speed > speed_of_light:
                        continue

                    # Concatenate the row to the existing DataFrame
                    lightning_strikes[j] = pd.concat([lightning_strikes[j], row], ignore_index=True)

                    # Update the time boundaries for this strike
                    strike_times[j] = (min(min_time, time1), max(max_time, time1))

                    data_found = True
                    break

            # If the row didn't match any existing strike, start a new one
            if not data_found:
                lightning_strikes.append(row)
                strike_times.append((time1, time1))  # Initialize min and max time with time1

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
