import os
from datetime import datetime
import pandas as pd  # Pands for csv creation and output
from typing import (
    Dict,
    Callable,
    Any,
    Tuple,
    List,
)  # For explicit types to rigid-ify the coding process

lightning_data_folder = "lightning_data"

# i.e. LYLOUT_240911_155000_0600.dat
data_extension = ".dat"

# i.e. Data: time (UT sec of day), lat, lon, alt(m), reduced chi^2, P(dBW), mask
data_header_startswith = "Data:"
data_header_startswith_len = len(data_header_startswith)

# i.e. *** data ***
data_body_start = "*** data ***"

lightning_data_output_folder = "lightning_data_output"

# Get all file names
file_names = os.listdir(lightning_data_folder)


def str_to_float(my_str: str) -> float:
    """
    :param my_str: The string (i.e. `"3.12"`)
    :return: The float representing the hex (`3.12`)
    """
    return float(my_str)


def str_hex_to_int(hex_str: str) -> int:
    """
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
process_handling: Dict[str, Callable[[str], Any]] = {
    "time (UT sec of day)": str_to_float,
    "lat": str_to_float,
    "lon": str_to_float,
    "alt(m)": str_to_float,
    "reduced chi^2": str_to_float,
    "P(dBW)": str_to_float,
    "mask": str_hex_to_int
}


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
filters: List = [
    ["reduced chi^2", accept_chi_below],
    ["alt(m)", data_above_20_km]
]


def mask_count_rule(n: int) -> bool:
    return n > 1

count_required = [
    ["mask", mask_count_rule]
]


def main():
    # Go through all names
    for file_name in file_names:

        # If does not end in data_extension (i.e. ".dat"), ignore
        if not file_name.endswith(data_extension):
            continue
        
        # Example: file_name = "LYLOUT_240911_155000_0600.dat"
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
        data_result: pd.DataFrame = None
        with open(file_path, "r") as f:
            data_result: pd.DataFrame = parse_file(f, month, day, year)

        # Output as csv format
        output_filename = os.path.splitext(file_name)[0] + ".csv"
        output_file = os.path.join(lightning_data_output_folder, output_filename)

        # Save as csv
        data_result.to_csv(output_file)

        # Note: You can retreive the csv 1:1 back as a pandas dataframe via: df = pd.read_csv('foo.csv')

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
    
    print("Tracked repeated occurances:", counters)

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
