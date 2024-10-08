import pandas as pd
import os
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import colorsys, pickle
from bmi_topography import Topography
import geopandas as gpd
import rasterio
import fsspec
import math
from typing import (
    Dict,
    Callable,
    Any,
    Tuple,
    List,
)  # For explicit types to rigid-ify the coding process


params = Topography.DEFAULT.copy()
params["south"] = 25.84  # Modify these coordinates for your area of interest (Texas)
params["north"] = 36.50
params["west"] = -106.65
params["east"] = -93.51

# Essential functions to use
count_required = []
"""
This determines count instances rule.\n
That is, if you provide a header, you provide a function for the count of items.\n
For example, for `1 >= n >= 2` that requires instances to be between `1` and `2` occurances.\n
Example:\n
```
count_required = [
    ["mask", mask_count_rule]
]
```
"""

process_handling: Dict[str, Callable[[str], Any]] = {}
"""
Callback functions for processing based on header, for translation\n

Dict[str, Callable[[str], Any]] \n
Basically means we are explicitly defining a dictionary named `process_handling`
with the key being a string `str`, and the value being a Callable object 
`Callable[[str], Any]` (basically a function). The callable function must have the 
parameter be a string but the return value can be anything.
Example:\n

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

# i.e. *** data ***
data_body_start = "*** data ***"

######################################################################################################
## Helper functions below with processing and retreiving a nice-looking DataFrame
######################################################################################################
def parse_file(f, month: int, day: int, year: int) -> pd.DataFrame:
    """
    This processes the entire file and extracts a pandas DataFrame

    :param f: The file object to read from
    :param month: An integer [1-12] representing the month
    :param day: An integer [1-31] representing the day
    :param year: An integer [1-9999] representing the year
    :return pd.DataFrame: A pandas dataframe resembling the data
    """
    # Data from the file
    data_headers: list[str] = None
    data_result: pd.DataFrame = None

    for line in f:  # Iterate through each line in the file
        line: str  # Hint that line is a string

        # Extract data headers if not found
        if not data_headers:
            if line.startswith(data_header_startswith):
                data_headers = return_data_headers_if_found(
                    line, data_header_startswith
                )

        # If headers are found, then we go through
        elif data_result == None:
            if line.strip() == data_body_start:
                data_result = parse_data(f, data_headers, month, day, year)

        # Assume fully indented the data and break the for loop
        else:
            break

    return data_result


def parse_data(
    f, data_headers: list[str], month: int, day: int, year: int
) -> pd.DataFrame:
    """
    This goes through the remaining of a file and processes the entire data into something a lot easier to use in Python

    :param f: The file, at the immediate point of data begin
    :param data_headers: The headers to compile to
    :param month: An integer [1-12] representing the month
    :param day: An integer [1-31] representing the day
    :param year: An integer [1-9999] representing the year
    :return pd.DataFrame: A pandas dataframe resembling the data
    """

    # Make dictionary (explicitly define that the key is a string and value is a list)
    dict_result: Dict[str, list] = {}

    dict_result["year"] = []
    dict_result["month"] = []
    dict_result["day"] = []

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

        # Process and format from string to respectable type
        allow_pass = True
        for i in range(len(data_row)):
            data_cell = data_row[i]

            # Format to respectable string
            if data_headers[i] in process_handling.keys():
                data_cell = process_handling[data_headers[i]](
                    data_cell
                )  # Process the data and parse it to designated format

            # Increment counter for the designated header (this is for counter filtering)
            if data_headers[i] in counters.keys():
                if not data_cell in counters[data_headers[i]].keys():
                    counters[data_headers[i]][data_cell] = 0
                
                counters[data_headers[i]][data_cell] += 1

            # Ensure that it's within filters rules. If it's not then prevent padding and break
            for j in range(len(filters)):
                if filters[j][0] == data_headers[i] and not filters[j][1](data_cell): # If the header name is the same (0 index)
                    allow_pass = False #Use as flag for allow_pass
                    break

            # Break if allow_pass flag is false
            if not allow_pass:
                break


            data_row[i] = data_cell

        # Skip the line and do not allow parsing if the allow_pass is false
        if not allow_pass:
            continue

        # Append to dictionary
        for i in range(len(data_row)):
            dict_result[data_headers[i]].append(
                data_row[i]
            )  # Add data cell to the designated region

        dict_result["month"].append(month)
        dict_result["day"].append(day)
        dict_result["year"].append(year)

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
    # Example: file_name = "LYLOUT_240911_155000_0600.dat"
    # Creating datetime of the file name using datetime object
    date_str = file_name[7:13]  # "240911" -> Sept 11, 2024
    time_str = file_name[14:20]  # "155000" -> 15:50:00
    dt_str = f"20{date_str} {time_str}"  # Adds "20" to the year to make it "2024"
    dt_format = "%Y%m%d %H%M%S"  # Formatting for it
    dt = datetime.strptime(dt_str, dt_format)

    # Obtain the file path relative to project directory to open
    file_path = os.path.join(lightning_data_folder, file_name)

    # Parse through data and retreive the Pandas DataFrame
    data_result: pd.DataFrame = None
    with open(file_path, "r") as f:
        data_result = parse_file(f, dt.month, dt.day, dt.year)
    return data_result


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

def cache_topography_data(pickle_file, params):
    # Check if the pickle file already exists
    if os.path.exists(pickle_file):
        print(f"Loading data from cache: {pickle_file}")
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    else:
        # Fetch the topography data if the pickle file does not exist
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
            tile_data = pickle.load(f)
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


# Define chunk size and over-extension
chunk_size = 2.0

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

downsampling_factor = 10  # Using every 5th point for downsampling

def plot_interactive_3d(df: pd.DataFrame, identifier: str, do_topography=True):
    """
    Create an interactive 3D scatter plot for latitude, longitude, and altitude, colored by mask.

    :param df: DataFrame containing lat, long, alt, and mask data
    :param identifier: Column to color the plot based on
    :return: Plotly figure object
    """

    # Topography generation
    # Topography generation
    # Generate a unique filename based on the bounding box

    os.makedirs("topography_cache", exist_ok=True)  # Ensure cache directory exists

    fig = go.Figure()

    if do_topography:
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

            fig.add_trace(go.Surface(
                z=elevation_data_downsampled,
                x=longitudes_downsampled,
                y=latitudes_downsampled,
                colorscale='Viridis',
                opacity=0.7,
                showscale=True
            ))


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

    fig.update_layout(
    scene=dict(
        xaxis=dict(
            title='Longitude',
            gridcolor='lightgray',  # Lighter color for gridlines
            zerolinecolor='lightgray',  # Lighter color for axis lines
            range=[min(params['west'], np.min(subset['lon'])), max(params['east'], np.max(subset['lon']))]
        ),
        yaxis=dict(
            title='Latitude',
            gridcolor='lightgray',  # Lighter color for gridlines
            zerolinecolor='lightgray',  # Lighter color for axis lines
            range=[min(params['south'], np.min(subset['lat'])), max(params['north'], np.max(subset['lon']))]
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