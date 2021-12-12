# Deep_NYC_Taxi_Bike

This is the repo for SUSTech Group Project, the deep learning research in NYC taxi and bike prediction.

## Dataset

The original data is from the public datasets of NYC taxi and bike. The datasets can be downloaded from the websites below.

**Bike:** [Citi Bike System Data | Citi Bike NYC](https://ride.citibikenyc.com/system-data)

**Taxi:** [TLC Trip Record Data - TLC (nyc.gov)](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

And we have done some data pre-processing for the Bike and Taxi datasets.

## Project Structure

The project structure is given as below

- data-NYCBike: The npz file for NYCBike datasets
- data-NYCTaxi: The npz file for NYCTaxi datasets
- data-processing: Generate the npy and npz files from the origin data
- model: Model and prediction for NYC data with STGCN, DCRNN and Graph-waveNet
- visualization: The prediction result for STGCN, DCRNN and Graph-waveNet, and some visualization



