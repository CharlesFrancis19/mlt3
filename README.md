# Bus Stop Prediction

Run `python bus_stop_prediction.py` to train a model and generate prediction CSV files for each stop.


The script uses `capymoa`'s `AdaptiveRandomForestRegressor` to fit a model for each

bus stop. You can provide a local dataset by setting the `BUS_DATASET_PATH`
environment variable. If not provided, the dataset is downloaded from
HuggingFace.

Predicted passenger counts are rounded to whole numbers for easier interpretation.

### Requirements

- pandas
- capymoa
- torch
- requests

