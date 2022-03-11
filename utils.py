import logging

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)
from sktime.forecasting.ets import AutoETS


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def run_for_one_intersection(df: pd.DataFrame, forecast_level: list, date_col: str,
                             dep_var: str, forecast_horizon: int):
    the_intersection = tuple(df[forecast_level].iloc[0])
    logger.info("Running for {}".format(the_intersection))

    result = pd.DataFrame()
    try:
        the_estimator = AutoETS(
            auto=True,
            maxiter=1000,
            additive_only=True,
            n_jobs=4,
        )

        fcst_horizon = np.arange(1, forecast_horizon + 1)

        the_estimator.fit(df[dep_var].astype("float64"))
        the_forecast = the_estimator.predict(fh=fcst_horizon)

        result = pd.DataFrame({
            dep_var: the_forecast
        })

        # insert forecast level columns
        for the_col in forecast_level:
            result.insert(0, the_col, df[the_col].unique()[0])

        # insert date column
        result.insert(0, date_col,
                      pd.date_range(start=df[date_col].max(), periods=forecast_horizon + 1, closed='right'))


    except Exception as e:
        logger.error("Error for intersection : {}".format(the_intersection))
        logger.exception(e)

    return result
