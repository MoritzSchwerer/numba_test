import polars as pl
import numba
import numpy as np
from tqdm import tqdm


MAPPING = {
    "column_1": "id",
    "column_2": "price",
    "column_3": "volume",
    "column_6": "datetime",
    "column_7": "isSell",
}

def format_data(data: pl.DataFrame) -> pl.DataFrame:
    data = data.rename(MAPPING).select(["id", "price", "volume", "datetime", "isSell"])
    data = data.with_column(pl.col('isSell').cast(pl.Int8))
    data = data.with_column((pl.col("datetime") * 1000).cast(pl.Datetime))
    return data


def get_data(files):
    combined = None
    for f in tqdm(files):
        data = pl.read_csv(f, has_header=False)
        data = format_data(data)
        if combined is None:
            combined = data
        else:
            combined = combined.extend(data)
    combined = combined.sort('id')
    return combined


def calculate_imbalance_bars(data:pl.DataFrame, threshold=1000) -> pl.DataFrame:
    volumes = data.select('volume').to_numpy().reshape(-1)
    sells = data.select('isSell').to_numpy().reshape(-1)
    imbalance_mask = get_imbalace_indices(volumes, sells, threshold)
    data = data.with_column(pl.Series(values=imbalance_mask, name='imbalance'))
    data = data.with_column(pl.col('imbalance').cast(pl.Int8).cumsum().alias('group'))
    bars = generate_bars(data)
    return bars
    
def calculate_imbalance_bars_numba(data:pl.DataFrame, threshold=1000) -> pl.DataFrame:
    volumes = data.select('volume').to_numpy().reshape(-1)
    sells = data.select('isSell').to_numpy().reshape(-1)
    imbalance_mask = get_imbalace_indices_numba(volumes, sells, threshold)
    data = data.with_column(pl.Series(values=imbalance_mask, name='imbalance'))
    data = data.with_column(pl.col('imbalance').cast(pl.Int8).cumsum().alias('group'))
    bars = generate_bars(data)
    return bars


def generate_bars(data: pl.DataFrame):
    ohlc = data.groupby('group').agg(
        [
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("volume").sum(),
            pl.count(),
            pl.col('id').first().alias('first_id'),
            pl.col('id').last().alias('last_id')
        ]
    )
    return ohlc

def get_imbalace_indices(volumes: np.ndarray, sells: np.ndarray, threshold: float) -> np.ndarray:
    new_bar = np.zeros(shape=(len(sells)), dtype=np.bool8)
    running_imb = 0
    for i in range(len(sells)):
        running_imb += -volumes[i] if sells[i] else volumes[i]
        if running_imb > threshold or running_imb < -threshold:
            running_imb = 0
            if i+1 < len(sells):
                new_bar[i+1] = 1
    return new_bar

@numba.njit()
def get_imbalace_indices_numba(volumes: np.ndarray, sells: np.ndarray, threshold: float) -> np.ndarray:
    new_bar = np.zeros(shape=(len(sells)), dtype=np.bool8)
    running_imb = 0
    for i in range(len(sells)):
        running_imb += -volumes[i] if sells[i] else volumes[i]
        if running_imb > threshold or running_imb < -threshold:
            running_imb = 0
            if i+1 < len(sells):
                new_bar[i+1] = 1
    return new_bar
         
    

