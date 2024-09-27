import os
import time
from datetime import datetime, date
import pickle as pkl
import pandas as pd
from typing import Dict, Callable, Any, Tuple

lightning_data_folder = "lightning_data"
data_extension = ".dat"

data_header_startswith = "Data:"
data_header_startswith_len = len(data_header_startswith)
data_body_start = "*** data ***"

lightning_data_output_folder = "lightning_data_output"

# Get all file names
file_names = os.listdir(lightning_data_folder)

def str_to_float(my_str:str) -> float:
    """
    :param my_str: The string (i.e. `"3.12"`)
    :return: The float representing the hex (`3.12`)
    """
    return float(my_str)

def str_hex_to_int(hex_str:str) -> int:
    """
    :param hex_str: The hex string (i.e. `"0x1721"`)
    :return: An integer representing the hex (`5921`)
    """
    return int(hex_str, 16)

# Callback functions for processing based on header, for translation
process_handling: Dict[str, Callable[[str], Any]] = {
    "time (UT sec of day)" : str_to_float,
    "lat" : str_to_float,
    "lon" : str_to_float, 
    "alt(m)" : str_to_float, 
    "reduced chi^2" : str_to_float,
    "P(dBW)" : str_to_float,
    "mask" : str_hex_to_int
}

def accept_chi_below_50(chi: float) -> bool:
    return chi <= 50.0

def data_above_20_km(alt_meters: float) -> bool:
    km = alt_meters / 1000.0 # Convert to kilometers
    return km >= 20.0

# After conversion process, you can now add callback functions for filters
# Accepts the designated row if the function returns true
filter_handling: Dict[str, Callable[[Any], bool]] = {
    "reduced chi^2" : accept_chi_below_50,
    "alt(m)" : data_above_20_km
}


def main():
    # Go through all names
    for file_name in file_names:

        # If does not end in data_extension (i.e. ".dat"), ignore
        if not file_name.endswith(data_extension):
            continue

        # Creating datetime of the file name using datetime object
        date_str = file_name[7:13]  # "240911" -> Sept 11, 2024
        time_str = file_name[14:20]  # "155000" -> 15:50:00
        dt_str = f"20{date_str} {time_str}"  # Adds "20" to the year to make it "2024"
        dt_format = "%Y%m%d %H%M%S"  # Formatting for it
        dt = datetime.strptime(dt_str, dt_format)

        file_path = os.path.join(lightning_data_folder, file_name)

        month = dt.month
        day = dt.day
        year = dt.year

        # Parse through data and retreive the Pandas DataFrame
        data_result:pd.DataFrame = None
        with open(file_path, "r") as f:
            data_result:pd.DataFrame  = parse_file(f, month, day, year)

        # Output as csv format
        output_filename = os.path.splitext(file_name)[0] + ".csv"
        output_file = os.path.join(lightning_data_output_folder, output_filename)

        # Save as csv
        data_result.to_csv(output_file)
        
        print(data_result)


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
        line: str # Hint that line is a string

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



def parse_data(f, data_headers: list[str], month: int, day: int, year: int) -> pd.DataFrame:
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

    # Make the keys for the dictionary with the designated headers
    for header in data_headers:
        dict_result[header] = []

    # Parse through the lines and apply
    for line in f:
        line: str # Hint that line is a string
        
        data_row = line.split() # Splits the line into designated sections

        # Process and format from string to respectable type
        allow_pass = True
        for i in range(len(data_row)):
            data_cell = data_row[i]

            # Format to respectable string
            if data_headers[i] in process_handling.keys():
                data_cell = process_handling[data_headers[i]](data_cell) # Process the data and parse it to designated format

            # Ensure that it's within filter bounds. If it's not then break
            if data_headers[i] in filter_handling.keys():
                if not filter_handling[data_headers[i]](data_cell):
                    allow_pass = False
                    break

            data_row[i] = data_cell

        if not allow_pass: # Skip the line and do not allow parsing
            continue
        
        # Append to dictionary
        for i in range(len(data_row)):
            dict_result[data_headers[i]].append(data_row[i]) # Add data cell to the designated region

        dict_result["month"].append(month)
        dict_result["day"].append(day)
        dict_result["year"].append(year)
            
    return pd.DataFrame(dict_result) # Return the item as a dataframe

def return_data_headers_if_found(
    line: str, data_header_startswith: str
) -> list[str] | None:
    """
    This parses a line and determines if the designated location is headers

    :param line: This is a single string to parse from (i.e. `"Data: hello world"`)
    :param data_header_startswith: This is a single string to identify where to start (i.e. `"Data:"`)
    :return list[str] | None: A list of headers (`list[str]`) or `None`
    """
    if line.startswith(data_header_startswith):
        # Remove the start name ("Data:")
        line = line[data_header_startswith_len:]
        data_headers = line.split(",")  # Extract all data headers

        # Clean up and remove the extra spacing if it exists
        for i in range(len(data_headers)):
            data_headers[i] = data_headers[i].strip()

        return data_headers
    return None


if __name__ == "__main__":
    main()
