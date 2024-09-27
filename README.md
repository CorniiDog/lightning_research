## Lightning Research Project

Connor White - Modular filtering algorithm, applied for space lightning

---

### Description

Instructions

1. Drag and drop the `LYLOUT_XXXXXX_XXXXXX_XXXX.dat` file into `lightning_data` folder
2. Run `main.py`
3. It will spit out `LYLOUT_XXXXXX_XXXXXX_XXXX.csv` in `lightning_data_output_folder`
4. You can now open up the .csv in any python application as a pandas dataframe (can be treated near-exactly like a dictionary) via  `df = pd.read_csv('path/to/LYLOUT_XXXXXX_XXXXXX_XXXX.csv')`

The CSV will now have the following headers:

`year	month	day	time (UT sec of day)	lat	lon	alt(m)	reduced chi^2	P(dBW)	mask`

---

```
git clone https://github.com/CorniiDog/lightning_research.git
```

Notables:

UT - seconds after midnight, in world time

About the naming scheme (i.e. lightning_data/LYLOUT_240911_235000_0600.dat):
>
> 240911 means 2024, sept 11
>
> 2350 - 23:50 or 11:50 in Z time
>
> 0600 means 600 seconds interval (or 10 mins)
> alt(m) is meters above mean sea level
> reduced chi ^2 - with TLE's, chi^2 of 50 or less get accepted, anything above gets rejected
>
> P is power (dbWatts)
>
> mask (an indicator of what is the same lightning flash) in hex
>
> task: find reliable source pts over 20 km


Prequisites:

```
sudo apt install gzip

pip install gdown pandas
```

To download the Sep 11, 2024 example lightning data: [reference](https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive)
```
cd lightning_data

gdown https://drive.google.com/uc?id=1EkYPqY0OmG5RBZH31Gb02hzey3c1Vxsx
```

Once you download a file, you can unzip via:

```
unzip "TLE WWLLN Data Zipped Sep272024.zip"
```


Next, to unzip all of the `.dat.gz` files to simply `.dat`:

```
gunzip *.gz
```

To run:

```
python main.py
```