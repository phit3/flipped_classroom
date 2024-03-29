# Flipped Classroom
> Reproduction package for: Flipped Classroom: Effective Teaching for Time Series Forecasting

## Preparing Run
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
    python3 main.py --tag essential --operation test --models TF --datasets lorenz --quiet
    ```
  - starting a test with all models and datasets of the 'exploratory' experiments.
    ```bash
    python3 main.py --tag exploratory --operation test --models TF FR CL_CTF_P CL_DTF_P_Lin CL_DTF_P_InvSig CL_DTF_P_Exp \
                    CL_DTF_D_Lin CL_DTF_D_InvSig CL_DTF_D_Exp CL_ITF_P_Lin CL_ITF_P_InvSig CL_ITF_P_Exp CL_ITF_D_Lin CL_ITF_D_InvSig CL_ITF_D_Exp \
                    --datasets thomas roessler lorenz lorenz96 --skip-missing --quiet
    ```
  - starting a test with all models and datasets of the 'essential' experiments.
    ```bash
    python3 main.py --tag essential --operation test --models TF FR CL_CTF_P CL_DTF_P_Lin CL_DTF_D_Lin CL_ITF_P_Lin CL_ITF_D_Lin \
                    --datasets mackeyglass thomas roessler hyperroessler lorenz lorenz96 --quiet
    ```
  - starting a training with TF on the mackeyglass and the roessler dataset while overriding some default hyperparameters (lr and plateau) saving weights tagged with 'the_future'
    ```bash
    python3 main.py --tag the_future --operation train --models TF --datasets \
                    '{"mackeyglass": {"lr": 1e-2}, "roessler": {"plateau": 30}}'
    ```  
  - starting a training using an increasing linear curriculum learning strategy (CL_ITF_P_Lin) for the three datasets lorenz, mackeyglass and roessler saving weights tagged with 'back_to_the_future'
    ```bash
    python3 main.py --tag back_to_the_future --operation train \
                    --models CL_ITF_P_Lin --datasets lorenz \
                    '{"mackeyglass": {"lr": 1e-2}, "roessler": {"plateau": 30}}'
    ```
