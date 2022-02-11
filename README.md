# Flipped Classroom
> Replication package for: Flipped Classroom: Effective Teaching for Chaotic Time Series Forecasting

## Preparing Run
### Setup Directory Structure
  - create data directory
    ```bash
    mkdir data
    ```
### Install Requirements
  - use pip3 to install required python3 libraries
    ```bash
    pip3 install -r requirements.txt
    ```
### Downloading Data
  - entering data directory
    ```bash
    cd data
    ```
  - download data from here: https://doi.org/10.7910/DVN/YEIZDT
  - unzip and put together the lorenz96N40 data parts
    ```bash
    unzip lorenz96N40_0.05_part1.csv.zip
    unzip lorenz96N40_0.05_part2.csv.zip
    cat lorenz96N40_0.05_part1.csv lorenz96N40_0.05_part2.csv > lorenz96_0.05.csv
    rm lorenz96N40_0.05_part*
    ```
  - rename some data files for compatibility reasons
    ```bash
    mv hyperroessler_0.1.csv hyperroessler_0.1_0.14.csv
    mv lorenz96_0.05.csv lorenz96_0.05_1.67.csv
    mv lorenz_0.01.csv lorenz_0.01_0.905.csv
    mv mackeyglass_1.0.csv mackeyglass_1.0_0.006.csv
    mv roessler_0.12.csv roessler0.12_0.069.csv
    mv thomas_0.1.csv thomas_0.1_0.055.csv
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
## Running Run
### In Times of Need
  - ask for help
    ```bash
    python3 main.py --help
    ```
### Examples
  - run test for teacher forcing (TF) trained model using weights saved under the tag 'essential', the lorenz' dataset
    ```bash
    python3 main.py --tag essential --operation test --models TF --datasets lorenz_0.01_0.905 --quiet
    ```
  - run training of model with teacher forcing (TF) on the mackeyglass and the roessler dataset while overriding some default hyperparameters saving weights under tag 'the_future'
    ```bash
    python3 main.py --tag essential --operation train --models TF --datasets \\
            '{"mackeyglass_1.0_0.006": {"lr": 1e-2}, "roessler_0.12_0.069": {"plateau": 30}}' \\
        --tag the_future
    ```  
  - run training of model using linear curriculum learning with increasing teacher forcing (CL_ITF_P_Lin) for the three datasets used above saving weights under tag 'back_to_the_future'
    ```bash
    python3 main.py --tag essential --operation train --models CL_ITF_P_Lin --datasets \\
         lorenz_0.01_0.905 \\
         '{"mackeyglass_1.0_0.006": {"lr": 1e-2}, "roessler_0.12_0.069": {"plateau": 30}}' \\
        --tag back_to_the_future
    ```