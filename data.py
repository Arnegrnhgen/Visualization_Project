import polars as pl
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyarrow import parquet as pq
import numpy as np


def load_parquet_data(property_path, start_dt, stop_dt):
    """
    Parameters
    ----------
    property_path : Path, [Path]
        path to doocs property paths. That directory should contain the paruqet files
    start_dt : datetime.datetime
        first datetime
    stop_dt : datetime.datetime
        last datetime

    Returns
    -------
    dictionary
        dictionary containing the requested data, a key for every doocs property path.
    """
    start_timestamp = datetime.timestamp(start_dt)
    stop_timestamp = datetime.timestamp(stop_dt)
    
    required_months = []
    tmp_dt = datetime(start_dt.year, start_dt.month, 1)
    while tmp_dt < stop_dt:
        file_path = tmp_dt.strftime("%Y-%m")
        required_months.append(file_path)
        tmp_dt += relativedelta(months=1)
    
    if not isinstance(property_path, list):
        property_path = [property_path]

    parquet_data = {}
    for p in property_path:
        parquet_files = [p.joinpath(f"{i}.parquet") for i in required_months]
        available_files = [i for i in parquet_files if i.is_file()]
        pq_dataset = pq.ParquetDataset(available_files, filters=[('timestamp', '>=', start_timestamp),('timestamp', '<=', stop_timestamp)])   
        
        parquet_data[p] = pq_dataset.read()
    return parquet_data


def get_doocs_properties(base_path):
    properties2paths = {}
    if base_path.is_dir():
        for fac_path in base_path.iterdir():
            fac = fac_path.name
            for dev_path in fac_path.iterdir():
                dev = dev_path.name
                for loc_path in dev_path.iterdir():
                    loc = loc_path.name
                    for prop_path in loc_path.iterdir():
                        prop = prop_path.name
                        properties2paths[str(prop_path)] = f"{fac}/{dev}/{loc}/{prop}"
    return properties2paths    


    


if __name__ == "__main__":

    base_path = Path("C:/Users/Arne/Sources/Ml_data/daqdata/sorted")
    properties = get_doocs_properties(base_path=base_path)
    properties

    # test_startdt = datetime(2023, 10, 15, 17, 30)
    # test_stopdt = datetime(2023, 11, 15, 17, 30)

    # test_path1 = Path("C:/Users/Arne/Sources/Ml_data/daqdata/sorted/XFEL.SYNC/LASER.LOCK.XLO/XTIN.MLO1/CURRENT_INPUT_JITTER.RD")
    # test_path2 = Path("C:/Users/Arne/Sources/Ml_data/daqdata/sorted/XFEL.SYNC/LASER.LOCK.XLO/XTIN.MLO1/CTRL0.OUT.STD_DEV.RD")

    # test_path = [test_path1, test_path2]

    # tic = datetime.now()
    # data = load_parquet_data(test_path, test_startdt, test_stopdt)
    # print((datetime.now()-tic).total_seconds())


