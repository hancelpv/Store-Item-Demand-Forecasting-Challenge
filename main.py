import logging

logging.basicConfig(level=logging.INFO, filename="out.log")
logger = logging.getLogger(__name__)
from utils import read_yaml
import pandas as pd
from joblib import Parallel, delayed
from utils import run_for_one_intersection
from tqdm_wrapper import tqdm_joblib
from tqdm import tqdm
import numpy as np


def main(config_path):
    try:
        logger.info("Inside main ...")
        logger.info("Reading config")

        config = read_yaml(config_path)

        data_config = config['data']
        forecast_config = config['forecast']

        logger.info("data_config : {}".format(data_config))
        logger.info("forecast_config : {}".format(forecast_config))

        # load train data
        train = pd.read_csv(data_config['data_path'] + "train.csv")
        test = pd.read_csv(data_config['data_path'] + "test.csv")
        sample_submission = pd.read_csv(data_config['data_path'] + "sample_submission.csv")

        # load configurables
        forecast_level = forecast_config['forecast_level']
        date_col = forecast_config['date_col']
        dep_var = forecast_config['dep_var']
        forecast_horizon = forecast_config['forecast_horizon']
        id_col = forecast_config['id_col']
        train[date_col] = pd.to_datetime(train[date_col])

        # cap negative values
        train[dep_var] = np.where(train[dep_var] < 0, 0, train[dep_var])

        with tqdm_joblib(tqdm(desc="My calculation", total=10)) as progress_bar:
            all_forecast_list = Parallel(n_jobs=4, verbose=1)(
                delayed(run_for_one_intersection)(
                    group,
                    forecast_level,
                    date_col,
                    dep_var,
                    forecast_horizon
                )
                for name, group in train.groupby(forecast_level)
            )

        all_forecasts = pd.DataFrame()
        if all_forecast_list:
            all_forecasts = pd.concat(all_forecast_list, ignore_index=True)

            test[date_col] = pd.to_datetime(test[date_col])

            # join on test dataframe to get id
            all_forecasts = all_forecasts.merge(test, on=forecast_level + [date_col])

            # filter only the relevant columns
            submission_df = all_forecasts[[id_col, dep_var]]

            # export as csv
            submission_df.to_csv(data_config["output_path"] + "submission.csv", index=False)

        logger.info("Finished execution ...")
    except Exception as e:
        logger.exception(e)
    return


if __name__ == '__main__':
    config_path = 'config.yaml'
    main(config_path)
