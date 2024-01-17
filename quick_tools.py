import os
import numpy as np
import pandas as pd


class CDataReader(object):
    def __init__(self, return_file_name: str, return_file_dir: str):
        self.return_file_name = return_file_name
        self.return_file_dir = return_file_dir
        self.return_df = pd.read_csv(self.return_file_path, dtype={"trade_date": str}).set_index("trade_date")

    @property
    def return_file_path(self) -> str:
        return os.path.join(self.return_file_dir, self.return_file_name)

    def get_range(self, bgn_date: str, end_date: str, normalize: bool, fillna_value: float = 0) -> pd.DataFrame:
        df = self.return_df.truncate(before=bgn_date, after=end_date).dropna(axis=1, how="all")
        if normalize:
            return ((df - df.mean()) / df.std()).fillna(fillna_value)
        return df.fillna(fillna_value)


def __mtm_diff(input_ret_df: pd.DataFrame, w: int, delay: int):
    mtm_df = input_ret_df.rolling(window=w).sum().shift(delay)
    test_df = pd.merge(
        left=input_ret_df, right=mtm_df,
        left_index=True, right_index=True,
        suffixes=("_e", "_r")
    ).dropna(axis=0, how="any")
    rank_ic = test_df.apply(
        lambda z: pd.DataFrame({'e': [z['a_e'], z['b_e']], 'r': [z['a_r'], z['b_r']]}).corr(
            method="spearman").at['e', 'r'],
        axis=1
    )
    ic_mean = rank_ic.mean()
    ic_std = rank_ic.std()
    icir = ic_mean / ic_std * np.sqrt(250 / w)
    return {
        "w": w,
        "delay": delay,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": icir,
    }


def mtm_diffs(input_ret_df: pd.DataFrame, ws: list[int], delay: int):
    res = []
    for w in ws:
        res.append(__mtm_diff(input_ret_df, w, delay))
    res_df = pd.DataFrame(res)
    print(res_df)
    return 0


def __mtm_simu(input_ret_df: pd.DataFrame, w: int, delay: int):
    mtm_df = input_ret_df.rolling(window=w).sum().shift(delay)
    test_df = pd.merge(
        left=input_ret_df, right=mtm_df,
        left_index=True, right_index=True,
        suffixes=("_e", "_r")
    ).dropna(axis=0, how="any")
    simu_ret = test_df.apply(
        lambda z: np.sign(z["a_e"] - z["b_e"]) * (z["a_r"] - z["b_r"]),
        axis=1
    )
    ret_mean = simu_ret.mean()
    ret_std = simu_ret.std()
    sharpe = ret_mean / ret_std * np.sqrt(250)
    return {
        "w": w,
        "delay": delay,
        "ret_mean": ret_mean,
        "ret_std": ret_std,
        "sharpe": sharpe,
    }


def mtm_simus(input_ret_df: pd.DataFrame, ws: list[int], delay: int):
    res = []
    for w in ws:
        res.append(__mtm_simu(input_ret_df, w, delay))
    res_df = pd.DataFrame(res)
    print(res_df)
    return 0


def quick_simu(df: pd.DataFrame, cost_rate: float):
    df["ret"] = df["signal"] * df["diff"]
    df["retCum"] = df["ret"].cumsum()
    df["cost"] = [1 * cost_rate if d0 != d1 else 0 for (d0, d1) in
                  zip(df["signal"], df["signal"].shift(1))]
    df["netRet"] = df["ret"] - df["cost"]
    df["netCum"] = df["netRet"].cumsum()
    return 0
