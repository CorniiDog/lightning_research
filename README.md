## Lightning Research Project

Connor White - Modular filtering algorithm, applied for space lightning

---

Prequisites:

```
sudo apt install gzip 

pip install gdown pandas PyQt5 matplotlib plotly streamlit
```

## Instructions

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

4. Next, to unzip all of the `.dat.gz` files (inside of lightning_data folder) to turn to simply `.dat`:

```
gunzip *.gz
```

7. Then, to run, do `streamlit run main.py`