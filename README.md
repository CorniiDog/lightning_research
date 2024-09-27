## Lightning Research Project

Connor White - Modular filtering algorithm, applied for space lightning

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