soil_moist_forecast
==============================

This is an end to end ds project for forecasting the soil moisture by reading iot data from sensors.
The structure of the project is based on the "cookiecutter" which is one of the based project structure for
data science projects.

General Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

The main notebooks are located in the 'notebook' directory. There are two Jupyter notebooks named 'soil_moist_forecast_draft' and 'soil_moist_forecast_final'. From a computational standpoint, both notebooks perform the same computations. The 'draft' version is the primary development notebook and does not have a concrete structure. However, in the final version, the code is organized as functions in the 'src' directory. I've kept the draft version in case your IDE encounters issues with the Python PATH, allowing you to run the code in the draft version.

The project involves conducting Exploratory Data Analysis (EDA) on soil moisture data from various locations in Europe. In the second part, an LSTM model is created to forecast the soil moisture for the next day by considering the measurements from the last 10 days.

TODOs:
The model can be deployed by FastAPI and after becomes dockerized can be used by an API call and predict the moisture for new incoming samples.
