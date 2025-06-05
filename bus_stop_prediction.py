import os
from io import StringIO
import pandas as pd
import requests
from capymoa.regressor import FIMTDD
from capymoa.stream import NumpyStream
from capymoa.evaluation import prequential_evaluation


DATA_URL = "https://huggingface.co/datasets/labiaufba/SSA_StopBusTimeSeries_5/raw/main/loader_03-05_2024.csv"


def load_dataset(path: str | None = None, url: str = DATA_URL) -> pd.DataFrame:
    """Load the CSV dataset from a local path or download it from HuggingFace."""
    if path and os.path.exists(path):
        df = pd.read_csv(path)
    else:
        resp = requests.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to download dataset: {resp.status_code}")
        df = pd.read_csv(StringIO(resp.text))
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def preprocess_stop_data(df: pd.DataFrame, stop_name: str, stop_id: int) -> pd.DataFrame:
    """Add time features and cleanup the dataframe for a single stop."""
    df_stop = df[["timestamp", stop_name]].dropna().rename(columns={stop_name: "actual"})
    df_stop["hour"] = df_stop["timestamp"].dt.hour
    df_stop["minute"] = df_stop["timestamp"].dt.minute
    df_stop["day"] = (df_stop["timestamp"] - df_stop["timestamp"].min()).dt.days
    df_stop["day_of_week"] = df_stop["timestamp"].dt.dayofweek
    df_stop["stop_id"] = stop_id
    # Store only the time-of-day string to remove the date component
    df_stop["timestamp"] = df_stop["timestamp"].dt.strftime("%H:%M:%S")
    return df_stop


def run_prediction_per_stop(df_stop: pd.DataFrame, stop_id: int) -> pd.DataFrame:
    """Train a FIMTDD model using CapyMOA and predict passenger counts."""
    features = ["hour", "minute", "day", "day_of_week", "stop_id"]

    X = df_stop[features].to_numpy()
    y = df_stop["actual"].to_numpy()

    stream = NumpyStream(X, y, target_type="numeric")
    learner = FIMTDD(stream.get_schema(), random_seed=42)

    results = prequential_evaluation(
        stream,
        learner,
        max_instances=len(X),
        store_predictions=True,
        store_y=True,
        restart_stream=False,
    )

    preds = pd.Series(results.predictions())
    rounded_pred = preds.round().astype(int)

    result_df = pd.DataFrame(
        {
            "timestamp": df_stop["timestamp"].values,
            "actual": results.ground_truth_y(),
            "predicted": rounded_pred,
            "error": (results.ground_truth_y() - rounded_pred).abs(),
        }
    )

    mae = results.cumulative.mae()
    print(f"    MAE for stop {stop_id}: {mae:.3f}")
    return result_df


def main() -> None:
    print("🚍 Starting bus stop prediction pipeline")
    dataset_path = os.environ.get("BUS_DATASET_PATH")
    df = load_dataset(dataset_path)
    stop_ids = df.columns[1:]

    for stop_name in stop_ids:
        stop_num = stop_ids.get_loc(stop_name)
        print(f"🔁 Processing stop: {stop_name} (ID: {stop_num})")
        df_stop = preprocess_stop_data(df, stop_name, stop_num)
        df_pred = run_prediction_per_stop(df_stop, stop_num)
        file_path = f"predictions_stop_{stop_num}.csv"
        df_pred.to_csv(file_path, index=False)
        print(f"✅ Saved {file_path}")


if __name__ == "__main__":
    main()
