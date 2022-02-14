# Flipped Classroom
> Replication package for: Flipped Classroom: Effective Teaching for Chaotic Time Series Forecasting

## Preparing Run
### Setting up Directory Structure
  - creating data directory
    ```bash
    mkdir data
    ```
### Installing Requirements
  - using pip3 to install required python3 libraries
    ```bash
    pip3 install -r requirements.txt
    ```
### Downloading Data
  - executing data download script
    ```bash
    python3 download_datasets.py
    ```
### Alternative: Generating Data
  - executing data generation script
    ```bash
    python3 generate_datasets.py
    ```
## Running Run
### In Times of Need
  - getting help
    ```bash
    python3 main.py --help
    ```
### Examples
  - starting a test the model trained with teacher forcing (TF) the lorenz' dataset using weights tagged with 'essential'.
    ```bash
    python3 main.py --tag essential --operation test --models TF --datasets lorenz_0.01_0.905 --quiet
    ```
  - starting a training with TF on the mackeyglass and the roessler dataset while overriding some default hyperparameters (lr and plateau) saving weights tagged with 'the_future'
    ```bash
    python3 main.py --tag the_future --operation train --models TF --datasets \
            '{"mackeyglass_1.0_0.006": {"lr": 1e-2}, "roessler_0.12_0.069": {"plateau": 30}}'
    ```  
  - starting a training using an increasing linear curriculum learning strategy (CL_ITF_P_Lin) for the three datasets lorenz, mackeyglass and roessler saving weights tagged with 'back_to_the_future'
    ```bash
    python3 main.py --tag back_to_the_future --operation train \
          --models CL_ITF_P_Lin --datasets lorenz_0.01_0.905 \
         '{"mackeyglass_1.0_0.006": {"lr": 1e-2}, "roessler_0.12_0.069": {"plateau": 30}}'
    ```