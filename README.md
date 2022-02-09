# Flipped Classroom
> Replication package for: Flipped Classroom: Effective Teaching for Chaotic Time Series Forecasting

## Preparing Run
### Setup Directory Structure
  - create data directory
    ```bash
    mkdir data
    mkdir checkpoint
    ```
### Downloading Data
  - entering data directory
    ````bash
    cd data
    ````
  - download data from here: https://doi.org/10.7910/DVN/YEIZDT
  - unzip and put together the lorenz96N40 data parts
    ```bash
    unzip lorenz96N40_0.05_part1.csv.zip
    unzip lorenz96N40_0.05_part2.csv.zip
    cat lorenz96N40_0.05_part1.csv lorenz96N40_0.05_part2.csv > lorenz96_0.05.csv
    rm lorenz96N40_0.05_part*
    ```
  - go back to root directory
    ```bash
    cd ..
    ```
### Alternative: Generating Data
  - executing data generation script
    ```bash
    python3 generate_datasets.py
    ```
### Install Requirements
  - use pip
    ```bash
    pip3 install -r requirements.txt
    ```
## Running Run
### In Times of Need
  - ask for help
    ```bash
    python3 main.py --help
    ```
### Examples
  - run test for TF using saved weights and lorenz' dataset
    ```bash
    python3 main.py --checkpoint  --strategy tf --data used_lorenz_0.01 --operation test
    ```