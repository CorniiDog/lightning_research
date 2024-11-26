## Lightning Research Project

Connor White - Modular filtering algorithm, applied for space lightning

This assumes that it will be running on a separate server, over an SSH session (ideally in VSCode).

Note: This system uses Google's Earth Engine as it is more extensive and robust than AWS's services. However, if the option arises that the opposite is true, I will add support for AWS.

---

## Prequisites

#### Quick Note

It is recommended to run the project in a virtual environment to avoid versioning conflicts.

1. [CTRL] + [SHIFT] + [P] (or [CMD] + [SHIFT] + [P] for MacOS) then "Python: Select Interpreter" then do "venv" and select a version (Recommended 3.10) to create a new virtual enviornment.
2. Restart the terminal for changes to take effect.

#### Instlaling G-Zip and other Python Packages 

1. Run the following commands:
```
sudo apt install gzip 

pip install gdown pandas PyQt5 matplotlib plotly streamlit bmi-topography geopandas fsspec requests aiohttp pyproj openpyxl streamlit-vis-timeline streamlit-chunk-file-uploader earthengine-api s3fs dask netcdf4 cartopy metpy imageio kaleido --no-cache-dir
```

#### Opentopography API

1. You need to get an OpenTopography API key here: https://opentopography.org/

2. When you get you API key, then you need to add the environment variable of the API key in your bashrc:
```
nano ~/.bashrc
```

3. Add to the bottom of the file:
```
export OPENTOPOGRAPHY_API_KEY=your_api_key
```
Replace `your_api_key` with the API key from the website

Then restart the terminal

#### Google's Earth Engine API

1. On your computer/laptop (locally, not on server), download and run `earthEngineAuthenticator.py` via `python3 earthEngineAuthenticator.py` with the `earthengine-api` package (`pip install earthengine-api`) installed. This should open a separate authentication window on your default browser to select an account to authenticate.

2. On the server (ssh that intends to run the data parser), make the earth engine directory via:
```
mkdir -p ~/.config/earthengine
```

3. On your computer/laptop (locally, not on server), copy the auto-generated credentials (`~/.config/earthengine/credentials`) from your local machine to the server with the data parser via:
```
scp ~/.config/earthengine/credentials user@remote_server:~/.config/earthengine/credentials 
```
Replace `user` with your ssh account username, and `remote_server` as traditionally the ip address of the server.


4. Verify that the credentials exist
On the client, the transfer should look similar to the following:
```
(base) connor@Connors-MacBook-Pro ~ % scp ~/.config/earthengine/credentials connor@100.104.180.24:~/.config/earthengine/credentials
credentials                                   100%  333    13.9KB/s   00:00    
(base) connor@Connors-MacBook-Pro ~ % 
```
On the server (ssh session):
```
(.conda) (base) connor@connor-server:~/.config/earthengine$ ls
credentials
(.conda) (base) connor@connor-server:~/.config/earthengine$ 
```

## Usage Instructions

1. Clone repository
```
git clone https://github.com/CorniiDog/lightning_research.git
```

2. Cd into the project folder, then (for instance) download the Sep 11, 2024 example lightning data: [reference](https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive)
```
cd lightning_data

gdown https://drive.google.com/uc?id=1EkYPqY0OmG5RBZH31Gb02hzey3c1Vxsx
```
3. Once you download a file, you can unzip (inside of lightning_data folder) via:

```
unzip "TLE WWLLN Data Zipped Sep272024.zip"
```

4. Next, to unzip all of the `.dat.gz` files (inside of lightning_data folder) to turn to simply `.dat` via the following command:

```
gunzip *.gz
```

7. Then, to run, do `streamlit run main.py`


## Credits/Contributions

- Topography data provided by OpenTopography: https://opentopography.org/
- Populated places (names with latitude and longitude of cities) provided by Natural Earth: https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-populated-places/
- Lightning data provided by Timothy Logan, Director of the Houston Lightning Mapping Array (HLMA) Network: https://artsci.tamu.edu/atmos-science/contact/profiles/timothy-logan.html


## Misc details:

Chunk file loader details: https://discuss.streamlit.io/t/new-component-chunk-file-uploader-break-through-the-upload-size-limit/61117

Streamlit vis timeline details: https://discuss.streamlit.io/t/new-component-streamlit-timeline-creating-beautiful-timelines-with-bi-directional-communication/31804