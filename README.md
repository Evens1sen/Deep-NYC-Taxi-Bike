# Deep_NYC_Taxi_Bike

This is the repo for the ITSC 2022 paper "Forecasting Regional Multimodal Transportation Demand with Graph Neural Networks: An Open Dataset!"
, the deep learning research in NYC taxi and bike prediction.

## Dataset

The original data is from the public datasets of NYC taxi and bike. The datasets can be downloaded from the websites below.

**Bike:** [Citi Bike System Data | Citi Bike NYC](https://ride.citibikenyc.com/system-data)

**Taxi:** [TLC Trip Record Data - TLC (nyc.gov)](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

And we have done some data pre-processing for the Bike and Taxi datasets.

### Summary of the public traffic dataset
| Graph-Based | Data Description / Data Source                               | Spatial Domain | Time Period         | Time Interval |
| :---------- | ------------------------------------------------------------ | :------------- | ------------------- | ------------- |
| METR_LA     | Traffic Speed Sensors in Los Angeles CountyLos Angeles Metropolitan Transportation Authority*Collaborated with University of Southern California  https://imsc.usc.edu/platforms/transdec/ | 207 sensors    | 2012/3/1∼2012/6/30  | 60 minutes    |
| BikeNYC     | Bike In-Out Flow / Bike Trip Data of New York City  https://www.citibikenyc.com/system-data | 69 regions     | 2019/1/1~2020/12/31 | 30/60 minutes |
| TaxiNYC     | Taxi In-Out Flow / Taxi Trip Data of New York City The New York City Taxi&Limousine Commission   (TLC) https://www1.nyc.gov/site/tlc/about/data.page | 69 regions     | 2019/1/1~2020/12/31 | 30/60 minutes |

## Project Structure

The project structure is given as below:

- data-NYCBike: The npz file for NYCBike datasets
- data-NYCTaxi: The npz file for NYCTaxi datasets
- data-NYCZones: The zones information for New York and the adjacency matrix
- data-processing: Generate the npy and npz files from the origin data
- model: Model and prediction for NYC data with STGCN, DCRNN and Graph-waveNet
- visualization: The prediction result for STGCN, DCRNN and Graph-waveNet, and some visualization






