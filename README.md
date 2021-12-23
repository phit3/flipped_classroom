# Flipped Classroom
Replication package for: Flipped Classroom: Effective Teaching for Chaotic Time Series Forecasting

## Preparing Run
### Downloading Data
  - create data directory
    ```bash
    mkdir data
    cd data
    ```
  - download data from here: https://doi.org/10.7910/DVN/YEIZDT
  - unzip and put together the lorenz96N40 data parts
    ```bash
    unzip lorenz96N40_0.05_part1.csv.zip
    unzip lorenz96N40_0.05_part2.csv.zip
    cat lorenz96N40_0.05_part1.csv lorenz96N40_0.05_part2.csv > lorenz96N40_0.05.csv
    rm lorenz96N40_0.05_part*
    ```
  - go back to root directory
    ```bash
    cd ..
    ```

### Install Requirements
  - Use pip
    ```bash
    pip3 install -r requirements.txt
    ```

## Running Run
  - Run main with help option
    ```bash
    python3 main.py --help
    ```
