import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import databento as db
import pandas as pd


API_KEY = os.getenv("DATABENTO_API_KEY", "db-xL9XiVKYxaPJSyXQnV7navu9VMVSV")
START_DATE = pd.Timestamp("2013-04-01")
END_DATE = pd.Timestamp("2024-12-31")
TIMEOUT_SEC = 45
RETRIES = 2

ROOT = Path("data") / "databento_raw"
DEF_DIR = ROOT / "definition_needed"
OHLCV_DIR = ROOT / "OHLCV-1d_needed"
LOG_DIR = ROOT / "logs"
for d in [DEF_DIR, OHLCV_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def call_with_timeout(fn, kwargs, timeout_sec):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, **kwargs)
        try:
            return fut.result(timeout=timeout_sec)
        except FuturesTimeout:
            fut.cancel()
            raise TimeoutError(f"request timed out after {timeout_sec}s")


def fetch_df(client, schema: str, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    kwargs = dict(
        dataset="OPRA.PILLAR",
        schema=schema,
        stype_in="parent",
        symbols=[symbol],
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    last_err = None
    for i in range(RETRIES):
        try:
            data = call_with_timeout(client.timeseries.get_range, kwargs, TIMEOUT_SEC)
            return data.to_df().reset_index()
        except Exception as e:
            last_err = e
            msg = str(e)
            if any(code in msg for code in ["401", "402", "403"]):
                raise
            if i < RETRIES - 1:
                time.sleep(2 * (i + 1))
    raise last_err


def build_needed_definition(def_df: pd.DataFrame, rebalance_date: pd.Timestamp, ticker: str) -> pd.DataFrame:
    if def_df.empty:
        return pd.DataFrame(
            columns=["date", "ticker", "cp_flag", "exp_date", "strike", "instrument_id", "dte"]
        )
    d = def_df.copy()
    if "expiration" not in d.columns or "instrument_class" not in d.columns or "strike_price" not in d.columns:
        return pd.DataFrame(
            columns=["date", "ticker", "cp_flag", "exp_date", "strike", "instrument_id", "dte"]
        )
    d["exp_date"] = pd.to_datetime(d["expiration"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    d["date"] = rebalance_date
    d["cp_flag"] = d["instrument_class"]
    d["strike"] = pd.to_numeric(d["strike_price"], errors="coerce")
    d["dte"] = (d["exp_date"] - d["date"]).dt.days
    out = d[
        (d["exp_date"] > d["date"])
        & d["dte"].between(21, 27, inclusive="both")
        & d["instrument_id"].notna()
    ].copy()
    out["ticker"] = ticker
    return out[["date", "ticker", "cp_flag", "exp_date", "strike", "instrument_id", "dte"]].drop_duplicates(
        ["date", "ticker", "cp_flag", "exp_date", "strike", "instrument_id"]
    )


def main():
    universe = pd.read_csv("UniverseTable.csv")
    if "rebalance_date" not in universe.columns or "ticker" not in universe.columns:
        raise ValueError("UniverseTable.csv must include rebalance_date and ticker")

    universe["rebalance_date"] = pd.to_datetime(universe["rebalance_date"], errors="coerce").dt.normalize()
    universe = universe.dropna(subset=["rebalance_date", "ticker"]).copy()
    universe = universe[
        (universe["rebalance_date"] >= START_DATE) & (universe["rebalance_date"] <= END_DATE)
    ]
    universe = universe[["rebalance_date", "ticker"]].drop_duplicates().sort_values(["rebalance_date", "ticker"])

    rows = universe.to_dict("records")
    client = db.Historical(API_KEY)
    failures = []

    print(f"Universe rows to process: {len(rows)}")
    for idx, row in enumerate(rows, start=1):
        rebalance_date = pd.Timestamp(row["rebalance_date"]).normalize()
        ticker = str(row["ticker"]).strip()
        symbol = f"{ticker}.OPT"

        def_file = DEF_DIR / f"{ticker}_{rebalance_date.strftime('%Y%m%d')}.parquet"
        ohlcv_file = OHLCV_DIR / f"{ticker}_{rebalance_date.strftime('%Y%m%d')}.parquet"

        if def_file.exists():
            needed_def = pd.read_parquet(def_file)
        else:
            try:
                raw_def = fetch_df(client, "definition", symbol, rebalance_date, min(rebalance_date + pd.Timedelta(days=1), END_DATE))
                needed_def = build_needed_definition(raw_def, rebalance_date, ticker)
                needed_def.to_parquet(def_file, index=False)
                print(f"[{idx}/{len(rows)}] saved needed definition {def_file.name} rows={len(needed_def)}")
            except Exception as e:
                failures.append(
                    {
                        "schema": "definition",
                        "ticker": ticker,
                        "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
                        "error": str(e),
                    }
                )
                print(f"[{idx}/{len(rows)}] failed needed definition {symbol} {rebalance_date.date()} err={e}")
                continue

        if needed_def.empty:
            if not ohlcv_file.exists():
                pd.DataFrame(columns=["date", "ticker", "cp_flag", "exp_date", "strike", "instrument_id", "mid"]).to_parquet(
                    ohlcv_file, index=False
                )
            continue

        needed_ids = set(pd.to_numeric(needed_def["instrument_id"], errors="coerce").dropna().astype("int64"))
        max_exp = pd.to_datetime(needed_def["exp_date"], errors="coerce").max()
        if pd.isna(max_exp):
            continue

        if ohlcv_file.exists():
            continue

        try:
            raw_ohlcv = fetch_df(client, "OHLCV-1d", symbol, rebalance_date, min(max_exp, END_DATE))
            if raw_ohlcv.empty:
                out = pd.DataFrame(columns=["date", "ticker", "cp_flag", "exp_date", "strike", "instrument_id", "mid"])
            else:
                tmp = raw_ohlcv.copy()
                tmp["instrument_id"] = pd.to_numeric(tmp["instrument_id"], errors="coerce").astype("Int64")
                tmp = tmp[tmp["instrument_id"].isin(list(needed_ids))].copy()
                if "symbol" in tmp.columns:
                    tmp["ticker"] = tmp["symbol"].astype(str).str.split(" ", n=1, expand=True)[0]
                else:
                    tmp["ticker"] = ticker
                if "ts_event" in tmp.columns:
                    tmp["date"] = pd.to_datetime(tmp["ts_event"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
                else:
                    tmp["date"] = rebalance_date
                tmp = tmp.merge(
                    needed_def[["instrument_id", "cp_flag", "exp_date", "strike"]].drop_duplicates("instrument_id"),
                    on="instrument_id",
                    how="left",
                )
                tmp["mid"] = pd.to_numeric(tmp["close"], errors="coerce")
                out = tmp[["date", "ticker", "cp_flag", "exp_date", "strike", "instrument_id", "mid"]].dropna(
                    subset=["date", "cp_flag", "exp_date", "strike", "instrument_id", "mid"]
                )
                out = out[(out["date"] >= rebalance_date) & (out["date"] <= out["exp_date"])].copy()
                out = out.sort_values(["date", "ticker", "cp_flag", "exp_date", "strike", "instrument_id"]).drop_duplicates(
                    ["date", "ticker", "cp_flag", "exp_date", "strike", "instrument_id"],
                    keep="last",
                )
            out.to_parquet(ohlcv_file, index=False)
            print(f"[{idx}/{len(rows)}] saved needed OHLCV {ohlcv_file.name} rows={len(out)}")
        except Exception as e:
            failures.append(
                {
                    "schema": "OHLCV-1d",
                    "ticker": ticker,
                    "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
                    "error": str(e),
                }
            )
            print(f"[{idx}/{len(rows)}] failed needed OHLCV {symbol} {rebalance_date.date()} err={e}")

    if failures:
        fail_df = pd.DataFrame(failures)
        out = LOG_DIR / "download_needed_failures.csv"
        if out.exists():
            prev = pd.read_csv(out)
            fail_df = pd.concat([prev, fail_df], ignore_index=True)
        fail_df = fail_df.drop_duplicates()
        fail_df.to_csv(out, index=False)
        print(f"Saved failures log: {out} rows={len(fail_df)}")
    else:
        print("All needed-only downloads completed without recorded failures.")


if __name__ == "__main__":
    main()
