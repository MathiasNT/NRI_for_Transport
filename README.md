# NRI for Transport <!-- omit in toc -->

Reposiory for the paper [Unboxing the graph: Neural Relational Inference for Mobility Prediction](https://arxiv.org/abs/2201.10307).

## Setup guide
1. Clone repo
2. Download data for the different experiments (or write to Write to mnity@dtu.dk for zipped versions of the data.)
   1. NYC Yellow Taxi data
      1. Download 2018 and 2019 data and shapefiles from https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.pag
   2. PEMS data
      1. Clone DCRNN repo and download PEMS data as specified here https://github.com/liyaguang/DCRNN
      2. Run their preprocessing as specified
      3. We need the files 
         1. ``adj_mx_bay.pkl``
         2. ``distances_bay_2017.csv``
         3. ``graph_sensor_locations_bay.csv``
         4. ``test.npz``
         5. ``train.npz``
         6. ``val.npz``
3. Setup a datafolder next to the github folder with the following folder structure

```
datafolder
├── procdata
│   ├── pems_data
│   └── taxi_data
│       └── full_manhattan
└── rawdata
    ├── pems
    │   ├── adj_mx_bay.pkl
    │   ├── distances_bay_2017.csv
    │   ├── graph_sensor_locations_bay.csv
    │   ├── test.npz
    │   ├── train.npz
    │   └── val.npz
    └── taxi
        ├── 2018
        │   ├── yellow_tripdata_2018-01.csv
        │   ├── yellow_tripdata_2018-02.csv
        │   ├── yellow_tripdata_2018-03.csv
        │   ├── yellow_tripdata_2018-04.csv
        │   ├── yellow_tripdata_2018-05.csv
        │   ├── yellow_tripdata_2018-06.csv
        │   ├── yellow_tripdata_2018-07.csv
        │   ├── yellow_tripdata_2018-08.csv
        │   ├── yellow_tripdata_2018-09.csv
        │   ├── yellow_tripdata_2018-10.csv
        │   ├── yellow_tripdata_2018-11.csv
        │   └── yellow_tripdata_2018-12.csv
        ├── 2019
        │   ├── yellow_tripdata_2019-01.csv
        │   ├── yellow_tripdata_2019-02.csv
        │   ├── yellow_tripdata_2019-03.csv
        │   ├── yellow_tripdata_2019-04.csv
        │   ├── yellow_tripdata_2019-05.csv
        │   ├── yellow_tripdata_2019-06.csv
        │   ├── yellow_tripdata_2019-07.csv
        │   ├── yellow_tripdata_2019-08.csv
        │   ├── yellow_tripdata_2019-09.csv
        │   ├── yellow_tripdata_2019-10.csv
        │   ├── yellow_tripdata_2019-11.csv
        │   └── yellow_tripdata_2019-12.csv
        └── shapefiles
            ├── taxi_zones.dbf
            ├── taxi_zones.prj
            ├── taxi_zones.sbn
            ├── taxi_zones.sbx
            ├── taxi_zones.shp
            ├── taxi_zones.shp.xml
            └── taxi_zones.shx
```
4. Run the preprocessing scripts (or alternatively the notebook in the notebook folder)
   1. ``NYC_taxi_preprocess_script.py``
   2. ``PEMS_data_preprocess_script.py``
5. Train models using the bash scripts in `bash_scripts`
   1. All bash scripts are set up with the hyperparameters from the paper 

Any inquiries feel free to contact mnity@dtu.dk.

