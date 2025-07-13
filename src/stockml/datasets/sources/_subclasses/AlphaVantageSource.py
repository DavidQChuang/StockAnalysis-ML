import calendar
import os
import re
from datetime import date, datetime, timedelta
from urllib.parse import urlencode

import pandas as pd
import pytz
from tqdm import tqdm

from ..Common import DatasourceConfig, Source


class AlphaVantageSource(Source):
    def _retrieve_dataframe(
        self, datasource_json: dict, config: DatasourceConfig, force_overwrite=False
    ) -> pd.DataFrame:
        url = "https://www.alphavantage.co/query?"

        if "alphavantage" not in datasource_json:
            raise Exception("'alphavantage' key must be present in dataset parameters.")

        query_params = datasource_json["alphavantage"]

        # Make sure apikey is present
        self._get_param(query_params, "apikey", required=True)

        # Recursively get all combinations of dataframe parameters and retrieve all needed CSVs
        # Get time in EST
        current_datetime = datetime.now(pytz.timezone("America/New_York"))
        current_date = current_datetime.date()

        # If the day hasn't ended yet, AlphaVantage hasn't updated for the current day.
        if current_datetime.hour < 16:
            current_date -= timedelta(days=1)

        dfs = []

        for url, filename in tqdm(
            self.get_urls(current_date, **query_params), "Downloading from AlphaVantage", ncols=80
        ):
            df = self.download_csv(url, filename, force_overwrite)
            dfs.append(df)

        # Combine the dfs, dropping original index
        df = pd.concat(dfs, ignore_index=True)

        # Drop index thing
        df.drop(df.columns[0], axis=1, inplace=True)

        # Convert timestamps to np.datetime64
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Sort by timestamp in ascending order
        df = df.sort_values(by="timestamp")

        # # Filter to standard hours
        # start_time = pd.to_datetime('09:30:00').time()
        # end_time = pd.to_datetime('16:00:00').time()
        # df = df[df['timestamp'].dt.time.between(start_time, end_time)] # type: ignore

        # Remove duplicates by timestamp
        df, old_df = df.drop_duplicates(subset=["timestamp"], keep="first"), df
        if len(old_df.values) != len(df.values):
            print(f"! Removing {len(old_df.values) - len(df.values)} duplicates")

        # Reset index and drop old
        df = df.reset_index(drop=True)

        return df

    # param getter helper
    def _get_param(self, query_params, param, required):
        if param in query_params:
            return query_params[param]
        elif not required:
            return ""
        else:
            raise Exception(f"'{param}' key must be present in AlphaVantageDatasource parameters.")

    def _process_multi_params(self, _param_name, _required, _current_date, **query_params):
        param = self._get_param(query_params, _param_name, _required)

        # if param is list, run get_dataframe_urls for each param
        if isinstance(param, list):
            # Get a copy of query_params ...
            query_params_copy = query_params.copy()
            urls = []

            # ... and replace the param list with a single value for each value in the list
            for each_value in param:
                query_params_copy[_param_name] = each_value
                urls += self.get_urls(_current_date, **query_params_copy)
            return urls

        # Return single param
        return param

    def get_urls(self, _current_date, **query_params):
        # Process multiparams
        function = self._process_multi_params("function", True, _current_date, **query_params)
        if isinstance(function, list):
            return function

        # If intraday process extra intraday parameters
        if function == "TIME_SERIES_INTRADAY":
            # Special case if month is '-[months]', generate a list of the past [months] months in YYYY-MM format.
            if "month" in query_params and isinstance(query_params["month"], str):
                month = query_params["month"]
                if month[0] == "-":
                    first_month = int(month)
                    query_params["month"] = [self.get_month(i, _current_date) for i in range(first_month, 1)]

            month = self._process_multi_params("month", False, _current_date, **query_params)
            if isinstance(month, list):
                return month

            interval = self._process_multi_params("interval", True, _current_date, **query_params)
            if isinstance(interval, list):
                return interval

        # Base case - get url & filename
        url = "https://www.alphavantage.co/query?"
        url += urlencode(query_params)

        # Used for file naming purposes, if the full month is present, YYYY-MM, else
        # YYYY-MM-DD is used if the month isn't over yet (i.e. the csv for that month will change in the future)
        month = query_params["month"] if "month" in query_params else self.get_month(0, _current_date)
        filename = self.get_filename(query_params, month, dir=function)

        return [(url, filename)]

    def get_month(self, offset: int, current_date: date):
        month = current_date.month
        year = current_date.year

        if offset != 0:
            while offset != 0:
                if offset < 0:
                    # Subtract one month for each offset
                    month -= 1
                    if month <= 0:
                        year -= 1
                        month = 12

                    offset += 1
                else:
                    # Add one month for each offset
                    month += 1
                    if month >= 13:
                        year += 1
                        month = 1

                    offset -= 1

            return "%d-%02d" % (year, month)
        else:
            last_day = calendar.monthrange(year, month)[1]

            # If this is the last day, use the normal string
            if current_date.day == last_day:
                return "%d-%02d" % (year, month)
            # Else add the day
            else:
                return "%d-%02d-%02d" % (year, month, current_date.day)

    def get_filename(self, query_params, month, dir):
        def get_param(param):
            return query_params.get(param, "")

        def param_name(param):
            return "-" + param if param else ""

        function = get_param("function")
        ticker = get_param("symbol")
        interval = get_param("interval")

        functions = {
            "TIME_SERIES_DAILY": "d",
            "TIME_SERIES_DAILY_ADJUSTED": "d",
            "TIME_SERIES_INTRADAY": "i",
            "DIGITAL_CURRENCY_DAILY": "dc-d",
        }

        # Add 'adj' and 'ext' to the end of the filename for adjusted and extended data
        if "adjusted" not in query_params or query_params["adjusted"]:
            month += "adj"

        if "extended" not in query_params or query_params["extended"]:
            month += "ext"

        # Shorten min to m
        interval = re.sub(r"([0-9]+)min", r"\1m", interval)

        if function in functions:
            function = functions[function]

        # If the file doesn't contain a whole month put it in a different folder
        if len(month.split("-")) == 3:
            dir = os.path.join(dir, "partial")

        return os.path.join("csv", dir, f"{function}{str(ticker)}{param_name(interval)}{param_name(month)}.csv")

    def download_csv(self, url: str, file_name: str, force_overwrite: bool = False):
        # timestamp,open,high,low,close,volume
        if force_overwrite or not os.path.exists(file_name):
            # print('Loading data from AlphaVantage url : %s' % url)
            df = pd.read_csv(url)

            if "close" not in df.columns and "close (USD)" not in df.columns:
                print(f"! Failed loading data from AlphaVantage url {url}: ")
                print("Columns: " + str(df.columns.values))
                raise FileNotFoundError(
                    f"Failed to retrieve data from AlphaVantage. You may have exceeded the free 5/min limit. {url}"
                )
            else:
                if not os.path.isdir("csv"):
                    os.mkdir("csv")

                idx = file_name.rfind("/")
                dir_name = file_name[:idx]

                if not os.path.isdir(dir_name):
                    os.mkdir(dir_name)

                # Standardize column names

                if "time" in df.columns:
                    df.rename(columns={"time": "timestamp"}, inplace=True)

                usd_cols_dict = {col: col.removesuffix(" (USD)") for col in df.columns if col.endswith("(USD)")}

                df.rename(columns=usd_cols_dict, inplace=True)
                df.to_csv(file_name)
        else:
            # print('Loading data from csv, original url : %s' % url)
            df = pd.read_csv(file_name)

        return df
