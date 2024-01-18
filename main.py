if __name__ == "__main__":
    import argparse
    import os
    import numpy as np
    import pandas as pd
    from husfort.qutility import make_twin_dir
    from quick_tools import CDataReader

    return_file_dir = r"E:\Deploy\Data\ForProjects\cta3\instruments_return"
    return_file_name = "instruments_return.csv.gz"
    save_root_dir = r"E:\ProjectsData"
    project_save_dir = make_twin_dir(save_root_dir, src=".")

    args_parser = argparse.ArgumentParser(description="Entry point of this project")
    args_parser.add_argument("--switch", type=str,
                             choices=("plot", "mtm", "simu", "auto", "ewm", "ewm-vol", "ewm-tnr"),
                             help="switch to functions")
    args_parser.add_argument("--bgn", type=str, help="format = [YYYYMMDD]")
    args_parser.add_argument("--end", type=str, help="format = [YYYYMMDD]")
    args_parser.add_argument("--delay", type=int, default=1, help="delay return in the future")
    args_parser.add_argument("-v", "--verbose", help="print details", action="store_true")
    args = args_parser.parse_args()

    bgn_date, end_date = args.bgn, args.end
    delay, verbose = args.delay, args.verbose

    data_reader = CDataReader(return_file_name, return_file_dir)
    df_ret = data_reader.get_range(bgn_date, end_date, normalize=False)
    if verbose:
        print(df_ret)

    groups = [
        ("A.DCE", "Y.DCE"),
        ("AG.SHF", "AU.SHF"),
        ("AL.SHF", "ZN.SHF"),
        ("BU.SHF", "TA.CZC"),
        ("C.DCE", "CS.DCE"),
        ("CU.SHF", "ZN.SHF"),
        ("HC.SHF", "RB.SHF"),
        ("I.DCE", "RB.SHF"),
        ("JM.DCE", "J.DCE"),
        ("L.DCE", "PP.DCE"),
        ("M.DCE", "RM.CZC"),
        ("MA.CZC", "V.DCE"),
        ("OI.CZC", "P.DCE"),
    ]
    ga, gb = zip(*groups)
    ga, gb = list(ga), list(gb)
    ga_ret = df_ret[ga]
    gb_ret = df_ret[gb]
    if verbose:
        print(ga_ret)
        print(gb_ret)

    input_ret_df = pd.DataFrame({'a': ga_ret.mean(axis=1), 'b': gb_ret.mean(axis=1)})
    input_nav_df = (input_ret_df + 1).cumprod()
    input_nav_df["diff"] = (input_ret_df['a'] - input_ret_df['b']) * 0.5
    input_nav_df["diffCum"] = input_nav_df["diff"].cumsum()
    input_nav_df.to_csv(os.path.join(project_save_dir, "nav_diff.csv"), float_format="%.8f")

    if args.switch == "plot":
        from husfort.qplot import CPlotLines, CPlotBars, CPlotLinesTwinxBar, CPlotScatter

        print(input_ret_df.corr())
        artist = CPlotBars(plot_df=input_ret_df, fig_save_dir=project_save_dir,
                           fig_name="pair_arbitrage_ret", fig_save_type="PNG")
        artist.plot()
        artist = CPlotLines(plot_df=input_nav_df["diffCum"], fig_save_dir=project_save_dir,
                            fig_name="pair_arbitrage_diff_cumsum", fig_save_type="PNG")
        artist.plot()
        artist = CPlotLinesTwinxBar(plot_df=input_nav_df, primary_cols=['a', 'b'], secondary_cols=['diff'],
                                    bar_color=["r"], fig_name="pair_arbitrage_nav", fig_save_type="PNG",
                                    fig_save_dir=project_save_dir)
        artist.plot()
        artist = CPlotScatter(plot_df=input_ret_df, point_x="a", point_y="b", fig_save_dir=project_save_dir,
                              fig_name="pair_arbitrage_scatter", fig_save_type="PNG")
        artist.plot()
    elif args.switch == "mtm":
        from quick_tools import mtm_diffs

        mtm_diffs(input_ret_df, ws=[1, 3, 5, 10, 20, 60], delay=delay)
    elif args.switch == "simu":
        from quick_tools import mtm_simus

        mtm_simus(input_ret_df, ws=[1, 3, 5, 10, 20, 60], delay=delay)
    elif args.switch == "auto":
        from quick_tools import quick_simu
        from husfort.qplot import CPlotLines

        cost_rate = 2e-4
        input_nav_df["diff_L1"] = input_nav_df["diff"].shift(1)
        input_nav_df["diff_L2"] = input_nav_df["diff"].shift(2)
        input_nav_df["diff_L3"] = input_nav_df["diff"].shift(3)
        input_nav_df["diff_L4"] = input_nav_df["diff"].shift(4)
        input_nav_df["diff_L5"] = input_nav_df["diff"].shift(5)
        auto_df = input_nav_df[["diff", "diff_L1", "diff_L2", "diff_L3", "diff_L4", "diff_L5"]].dropna(
            axis=0, how="any")
        acr = auto_df.corr()
        print(acr)

        input_nav_df["dir"] = np.sign(input_nav_df["diff"])
        input_nav_df["signal"] = input_nav_df["dir"].shift(delay).fillna(0)
        quick_simu(df=input_nav_df, cost_rate=cost_rate)
        artist = CPlotLines(
            plot_df=input_nav_df[["diffCum", "retCum", "netCum"]],
            fig_name=f"auto_D{delay}", line_color=["#483D8B", "#FF69B4", "#DC143C"],
            line_style=["-.", "--", "-"], line_width=1.0,
            fig_save_type="PNG", fig_save_dir=project_save_dir,
        )
        artist.plot()
    elif args.switch == "ewm":
        from husfort.qplot import CPlotLines
        from quick_tools import quick_simu, quick_report

        fast, slow = 0.9, 0.6
        cost_rate = 2e-4
        input_nav_df["MAFast"] = input_nav_df["diff"].ewm(alpha=fast, adjust=False).mean()
        input_nav_df["MASlow"] = input_nav_df["diff"].ewm(alpha=slow, adjust=False).mean()
        input_nav_df["dir"] = input_nav_df.apply(
            lambda z: 1 if z["MAFast"] >= z["MASlow"] else -1,
            axis=1
        )
        input_nav_df["signal"] = input_nav_df["dir"].shift(delay).fillna(0)
        quick_simu(df=input_nav_df, cost_rate=cost_rate)
        quick_report(input_nav_df, ret_ids=["ret", "netRet"])

        artist = CPlotLines(
            plot_df=input_nav_df[["diffCum", "retCum", "netCum"]],
            fig_name=f"ewm_D{delay}", line_color=["#483D8B", "#FF69B4", "#DC143C"],
            line_style=["-.", "--", "-"], line_width=1.0,
            fig_save_type="PNG", fig_save_dir=project_save_dir
        )
        artist.plot(ylim=(-0.2, 0.7))
    elif args.switch == "ewm-vol":
        from husfort.qplot import CPlotLines
        from quick_tools import quick_simu, quick_report


        def __get_direction_ewm_vol(fast_val: float, slow_val: float, volatility: float) -> int:
            if volatility > threshold:
                return 1 if fast_val >= slow_val else -1
            else:
                return 0


        fast, slow = 0.9, 0.6
        rolling_win = 20
        cost_rate = 1e-4
        threshold = 0.04

        input_nav_df["MAFast"] = input_nav_df["diff"].ewm(alpha=fast, adjust=False).mean()
        input_nav_df["MASlow"] = input_nav_df["diff"].ewm(alpha=slow, adjust=False).mean()
        input_nav_df["volatility"] = input_nav_df["diff"].rolling(window=rolling_win).std() * np.sqrt(250)
        input_nav_df["dir"] = input_nav_df.apply(
            lambda z: __get_direction_ewm_vol(z["MAFast"], z["MASlow"], z["volatility"]),
            axis=1
        )
        input_nav_df["signal"] = input_nav_df["dir"].shift(delay).fillna(0)
        quick_simu(df=input_nav_df, cost_rate=cost_rate)
        quick_report(input_nav_df, ret_ids=["ret", "netRet"])

        artist = CPlotLines(
            plot_df=input_nav_df[["diffCum", "retCum", "netCum"]],
            fig_name=f"ewm-vol_D{delay}", line_color=["#483D8B", "#FF69B4", "#DC143C"],
            line_style=["-.", "--", "-"], line_width=1.0,
            fig_save_type="PNG", fig_save_dir=project_save_dir
        )
        artist.plot(ylim=(-0.2, 0.7))
    elif args.switch == "ewm-tnr":
        from husfort.qplot import CPlotLines
        from quick_tools import quick_simu, quick_report


        def __get_direction_ewm_tnr(fast_val: float, slow_val: float, tnr: float) -> int:
            if tnr > threshold:
                return 1 if fast_val >= slow_val else -1
            else:
                return 0


        fast, slow = 0.9, 0.6
        rolling_win = 5
        cost_rate = 1e-4
        threshold = 0.05

        input_nav_df["MAFast"] = input_nav_df["diff"].ewm(alpha=fast, adjust=False).mean()
        input_nav_df["MASlow"] = input_nav_df["diff"].ewm(alpha=slow, adjust=False).mean()
        input_nav_df["rngSumAbs"] = input_nav_df["diff"].abs().rolling(window=rolling_win).sum()
        input_nav_df["rngAbsSum"] = input_nav_df["diff"].rolling(window=rolling_win).sum().abs()
        input_nav_df["tnr"] = input_nav_df["rngAbsSum"] / input_nav_df["rngSumAbs"]

        input_nav_df["dir"] = input_nav_df.apply(
            lambda z: __get_direction_ewm_tnr(z["MAFast"], z["MASlow"], z["tnr"]),
            axis=1
        )
        input_nav_df["signal"] = input_nav_df["dir"].shift(delay).fillna(0)
        quick_simu(df=input_nav_df, cost_rate=cost_rate)
        quick_report(input_nav_df, ret_ids=["ret", "netRet"])

        artist = CPlotLines(
            plot_df=input_nav_df[["diffCum", "retCum", "netCum"]],
            fig_name=f"ewm-tnr_D{delay}", line_color=["#483D8B", "#FF69B4", "#DC143C"],
            line_style=["-.", "--", "-"], line_width=1.0,
            fig_save_type="PNG", fig_save_dir=project_save_dir
        )
        artist.plot(ylim=(-0.2, 0.7))

    else:
        print(f"switch = {args.switch}")
        raise ValueError
