import datetime
from pathlib import Path
from volumeImbalanceBars import get_data, calculate_imbalance_bars, calculate_imbalance_bars_numba
import time





if __name__ == '__main__':
    path = Path("/home/moritz/datasets/crypto/raw/")

    files = [f for f in path.rglob("*.csv")]

    data = get_data(files)
    print("starting calculation.")

    print("No opt:")
    results = []
    for _ in range(2):
        start = time.time()
        bars = calculate_imbalance_bars(data, threshold=10000)
        end = time.time()
        results.append((end-start))
        print(results[-1], " s.")
    print(f"Took {round(sum(results)/len(results), 3)} on average.")

    print("\n", "="*40)
    print("Numba njit:")
    results = []
    for _ in range(10):
        start = time.time()
        bars = calculate_imbalance_bars_numba(data, threshold=10000)
        end = time.time()
        results.append((end-start))
        print(results[-1], " s.")
    print(f"Took {round(sum(results)/len(results), 3)} on average.")

