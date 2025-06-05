import os
from io import StringIO
import pandas as pd
import requests
from capymoa.stream import stream_from_file
from capymoa.regressor import AdaptiveRandomForestRegressor


DATA_URL = "https://huggingface.co/datasets/labiaufba/SSA_StopBusTimeSeries_5/raw/main/loader_03-05_2024.csv"


def download_dataset(url: str = DATA_URL) -> pd.DataFrame:
    """Download the CSV dataset from HuggingFace."""
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


def run_prediction_per_stop(df_stop: pd.DataFrame, stop_id: int, *, limit: int | None = 1000) -> pd.DataFrame:
    """Predict passenger counts for a single stop using a streaming model."""
    tmp_path = f"temp_stop_{stop_id}.csv"
    df_stop[["hour", "minute", "day", "day_of_week", "stop_id", "actual"]].to_csv(tmp_path, index=False)

    stream = stream_from_file(tmp_path, target_type="numeric")
    schema = stream.get_schema()
    model = AdaptiveRandomForestRegressor(schema=schema, ensemble_size=20)

    results = []
    for i, instance in enumerate(stream):
        if limit is not None and i >= limit:
            break
        pred = model.predict(instance) or 0.0
        model.train(instance)
        rounded_pred = round(pred)
        results.append({
            "timestamp": df_stop.iloc[i]["timestamp"],
            "actual": instance.y_value,
            "predicted": rounded_pred,
            "error": abs(instance.y_value - rounded_pred),
        })

    os.remove(tmp_path)
    return pd.DataFrame(results)


def main() -> None:
    print("🚍 Starting bus stop prediction pipeline")
    df = download_dataset()
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
