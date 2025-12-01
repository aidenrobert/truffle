"""
Welcome to TRUFFLE:
Trained Recognition of Unique Fluorescence- & Form-based Labels for Environmental aerosols.
This code is developed for use with Multiparameter Bioaerosol Spectrometer (MBS;
University of Hertfordshire, UK) data for particle classification and general utilities.
"""
__author__ = "Aiden Jönsson"
__copyright__ = "Stockholm University"
__credits__ = ["Aiden Jönsson", "Paul Zieger", "Gabriel Freitas"]
__license__ = "GPL-3.0"
__date__ = "2025-12-01"
__version__ = "0.1"
__maintainer__ = "Aiden Jönsson"
__emails__ =  "aiden.jonsson@aces.su.se"
__status__ = "Prototype"

def flag_pollution(
        df, return_probabilities=True,
        model_file="MBS_logreg_pollution_TUNED_251102.pkl",
        alpha=None
    ):

    """
    Classify particles as polluted (combustion-like) ONLY for rows where
    df['group'] is 'HFP' or 'PBAP'. Other rows are left unchanged.

    Returns a full DataFrame with 'poll_flag' and optionally 'poll_prob'.

    Parameters
    ----------
    df : pandas.DataFrame
        Full particle dataframe with required input fields.
    return_probabilities : bool, default=True
        Whether to return logistic regression model probabilities.
    model_file : str
        File name of trained logistic regression model.
    alpha : float
        Confidence level for positive pollution identification.

    Returns
    -------
    pandas.DataFrame
        Full dataframe including:
            poll_flag : integer {0, 1} or NaN
            poll_prob : float or NaN (optional)
    """

    import numpy as np
    import joblib
    df = df.copy()

    ## Work only on these groups
    mask = df["group"].isin(["HFP", "PBAP"])
    sub = df.loc[mask].copy()

    ## Create empty columns for the whole dataframe
    df["poll_flag"] = np.nan
    if return_probabilities:
        df["poll_prob"] = np.nan

    ## If nothing to classify, just return df
    if sub.empty:
        return df

    ## Construct saturation flag
    sub["Sat"] = (sub["saturated"] == "yes").astype(float)

    ## Predictor columns
    input_cols = [
        "XE1_1_norm", "XE1_2_norm", "XE1_3_norm", "XE1_4_norm",
        "XE1_5_norm", "XE1_6_norm", "XE1_7_norm", "XE1_8_norm",
        "Size", "FL_ratio", "FL", "AsymLR%", "AsymLRinv%",
        "PeakMeanR", "PeakWidthR", "PeakCountR", "KurtosisR",
        "VarianceR", "MeanR", "SumR", "SkewR", "MirrorR%",
        "PeakMeanL", "PeakWidthL", "PeakCountL", "KurtosisL",
        "VarianceL", "MeanL", "SumL", "SkewL", "MirrorL%",
        "Sat",
    ]

    ## Extract predictors
    x = sub[input_cols].to_numpy()

    ## Identify NaN rows
    bad_idx = np.isnan(x).any(axis=1)

    ## Replace NaNs and infinities
    fmax = np.finfo(np.float32).max
    fmin = np.finfo(np.float32).min
    x = np.nan_to_num(x, nan=0.0, posinf=fmax, neginf=fmin)

    ## Clip extreme values
    mask_clip = np.abs(x) > fmax
    x[mask_clip] = np.sign(x[mask_clip]) * fmax

    ## Predict
    logreg = joblib.load("models/" + model_file)
    probs = logreg.predict_proba(x)
    sub["poll_flag"] = np.argmax(probs, axis=1)
    if return_probabilities:
        sub["poll_prob"] = probs[:, 1]

    ## Handle rows with invalid input
    sub.loc[bad_idx, "poll_flag"] = 1
    if return_probabilities:
        sub.loc[bad_idx, "poll_prob"] = np.nan

    ## Insert results back into the original dataframe
    df.loc[sub.index, ["poll_flag"]] = sub["poll_flag"]
    if return_probabilities:
        df.loc[sub.index, ["poll_prob"]] = sub["poll_prob"]

    return df


def classify_fluo(df, return_probabilities=True, return_umap=True):
    
    import joblib
    import numpy as np

    """
    Classify particles as polluted (combustion-like) ONLY for rows where
    df['group'] is 'HFP' or 'PBAP'. Other rows are left unchanged.

    Returns a full DataFrame with UMAP and k-nearest neighbors-derived
    classification of fluorescent particles, with optional probabilities
    and UMAP dimension values.

    Output class indices:
    0: Cannot compute; used to denote where inputs make classification impossible
    1: Combustion
    2: Pollen
    3: Bacteria
    4: Fungal spore

    Parameters
    ----------
    df : pandas.DataFrame
        Full particle dataframe with required input fields.
    return_probabilities : bool, default=True
        Whether to return logistic regression model probabilities.
    model_file : str
        File name of trained logistic regression model.

    Returns
    -------
    pandas.DataFrame
        Full dataframe including:
            class : integer {0, 1} or NaN
            class_prob : float or NaN (optional)
            UMAP1 : float or NaN (optional)
            UMAP2 : float or NaN (optional)
    """

    ## Safe copy
    df = df.copy()
    hfp_df = df[df["group"].isin(["HFP", "PBAP"])].copy()

    ## Saturation flag
    hfp_df["Sat"] = (hfp_df["saturated"]=="yes").astype(float)

    ## Define and choose columns used as input for the models
    input_cols = [
        "XE1_1_norm", "XE1_2_norm", "XE1_3_norm", "XE1_4_norm",
        "XE1_5_norm", "XE1_6_norm", "XE1_7_norm", "XE1_8_norm",
        "Size", "FL_ratio", "FL", "AsymLR%", "AsymLRinv%",
        "PeakMeanR", "PeakWidthR", "PeakCountR", "KurtosisR",
        "VarianceR", "MeanR", "SumR", "SkewR", "MirrorR%",
        "PeakMeanL", "PeakWidthL", "PeakCountL", "KurtosisL",
        "VarianceL", "MeanL", "SumL", "SkewL", "MirrorL%",
        "Sat",
    ]
    x = hfp_df[input_cols].to_numpy()

    ## Replace NaN and infinities
    fmax = np.finfo(np.float32).max
    fmin = np.finfo(np.float32).min
    x = np.nan_to_num(x, nan=0.0, posinf=fmax, neginf=fmin)

    ## Clip overflowing values
    mask = np.abs(x) > fmax
    x[mask] = np.sign(x[mask]) * fmax

    ## Load UMAP approximator and kNN model
    approx = joblib.load("models/" + "MBS_umap_approximator_251102.pkl")
    knn = joblib.load("models/" + "MBS_umap_knnmodel_251102.pkl")

    ## Apply UMAP transformation
    umap = approx.transform(x)

    ## Replace NaN UMAP outputs; bad_idx is used to classify these as 0
    bad_idx = np.isnan(umap).any(axis=1)
    umap[bad_idx] = 0.0

    ## Predict class using kNN model
    hfp_df["class"] = knn.predict(umap).astype(int)
    #if return_prob:
    hfp_df["class_prob"] = knn.predict_proba(umap).max(axis=1)

    ## Add UMAP coordinates
    if return_umap : hfp_df.loc[:, ["UMAP1", "UMAP2"]] = umap

    ## Clear problematic points and assign them Class 0
    hfp_df.loc[bad_idx, ["class", "UMAP1", "UMAP2"]] = [0, np.nan, np.nan]
    if return_probabilities : hfp_df.loc[bad_idx, "class_prob"] = np.nan

    ## Return the subsetted fluorescent data back to the original
    cols = ["class"]
    if return_probabilities : cols += ["class_prob"]
    if return_umap : cols += ["UMAP1", "UMAP2"]
    for col in cols:
        if col not in df.columns : df[col] = np.nan
    df.loc[hfp_df.index, cols] = hfp_df[cols]

    return df

def flag_dust(
        df, return_probabilities=True,
        model_file="MBS_logreg_dust_TUNED_251121.pkl",
        alpha=None
    ):

    """
    Classify coarse particles (CP or FP groups, Size ≥ 2.5 µm) into
    dust (class 0) or sea-salt aerosol (class 1) using a trained
    logistic regression classifier.

    Classification is applied *only* to rows meeting:
        df["group"].isin(["CP", "FP"]) AND df["Size"] >= 2.5

    All other rows return NaN for 'dust_flag' (and 'dust_prob').

    Parameters
    ----------
    df : pandas.DataFrame
        Full particle dataframe with required input fields.
    return_probabilities : bool, default=True
        Whether to return class-1 probabilities.
    model_file : str
        File name of trained logistic regression model.
    alpha : float
        Confidence level for positive dust identification.

    Returns
    -------
    pandas.DataFrame
        Full dataframe including:
            dust_flag : integer {0, 1} or NaN
            dust_prob : float or NaN (optional)
    """

    import numpy as np
    import joblib

    df = df.copy()

    ## Initialize result columns on full dataframe
    df["dust_flag"] = np.nan
    if return_probabilities:
        df["dust_prob"] = np.nan

    ## Define subset for dust classification
    mask = df["group"].isin(["CP", "FP"]) & (df["Size"] >= 2.5)

    ## Nothing to classify → return immediately
    if not mask.any():
        return df
    sub = df.loc[mask].copy()

    ## Predictor columns
    cols = [
        "AsymLR%", "AsymLRinv%", 
        "PeakMeanR", "PeakWidthR", "PeakCountR", "KurtosisR",
        "VarianceR", "MeanR", "SumR", "SkewR", "MirrorR%",
        "PeakMeanL", "PeakWidthL", "PeakCountL", "KurtosisL",
        "VarianceL", "MeanL", "SumL", "SkewL", "MirrorL%",
    ]
    x = sub[cols].to_numpy()

    ## Identify bad rows (NaN anywhere)
    bad_idx = np.isnan(x).any(axis=1)

    ## Replace NaN and infinities
    fmax = np.finfo(np.float32).max
    fmin = np.finfo(np.float32).min
    x = np.nan_to_num(x, nan=0.0, posinf=fmax, neginf=fmin)

    ## Clip extreme values
    mask_clip = np.abs(x) > fmax
    x[mask_clip] = np.sign(x[mask_clip]) * fmax

    ## Predict
    logreg = joblib.load("models/" + model_file)
    probs = logreg.predict_proba(x)
    sub["dust_flag"] = np.argmax(probs, axis=1)
    if return_probabilities:
        sub["dust_prob"] = probs[:, 1]

    ## Override bad rows → NaN
    sub.loc[bad_idx, "dust_flag"] = np.nan
    if return_probabilities:
        sub.loc[bad_idx, "dust_prob"] = np.nan

    sub["dust_flag"] = sub["dust_flag"].astype("Int64")

    ## Merge back into full dataframe
    df.loc[sub.index, "dust_flag"] = sub["dust_flag"]
    if return_probabilities:
        df.loc[sub.index, "dust_prob"] = sub["dust_prob"]

    return df


def pig(
    df,
    return_poll_prob=False,
    return_class_prob=False,
    return_dust_prob=False,
    return_umap=False,
    alpha_poll=None,
    alpha_dust=None,
    clear=True
    ):

    """
    PIG: the Particle Identification Gadget
    Combines all models: the pollution model, fluorescent particle classifier, and dust model;
    and provides output giving best-guess classes and their probabilities.

    All rows where no classification is performed receives NaN in respective columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Full particle dataframe with required input fields.
    return_poll_prob : bool, default=True
        Whether to return pollution logistic regression model probabilities.
    return_dust_prob : bool, default=True
        Whether to return dust logistic regression model probabilities.
    return_class_prob : bool, default=True
        Whether to return k-nearest neighbor classifier probabilities.
    alpha_poll : float, default=None
        Confidence level for positive pollution LRM identification.
    alpha_dust : float, default=None
        Confidence level for positive dust LRM identification.

    Returns
    -------
    pandas.DataFrame
        Full dataframe including:
            dust_flag : integer {0, 1} or NaN
            dust_prob : float or NaN (optional)
    """

    from IPython.display import clear_output
    
    ## Safe copy
    df = df.copy()

    ## Run dust model
    print("Sniffing... 𓍊𓋼𓍊𓋼𓍊 ......... 𓃟")
    df = flag_dust(
        df,
        return_probabilities=return_dust_prob,
        alpha=alpha_dust
        )
    
    ## Run pollution model
    #clear_output(wait=True)
    print("Sniffing... 𓍊𓋼𓍊𓋼𓍊 ...... 𓃟")
    df = flag_pollution(df, return_poll_prob, alpha=alpha_poll)

    ## Run fluorescent particle classifier
    if clear : clear_output(wait=True)
    print("Sniffing... 𓍊𓋼𓍊𓋼𓍊 ... 𓃟")
    df = classify_fluo(
        df, 
        return_probabilities=return_class_prob,
        return_umap=return_umap
        )

    ## Create a PBAP flag
    hfp_index = df["group"].isin(["HFP", "PBAP"])
    df.loc[hfp_index, "PBAP_flag"] = 0
    bio_idx = (df["class"].isin([2, 3, 4])) & (df["poll_flag"]==0)
    df.loc[bio_idx, "PBAP_flag"] = 1

    if clear : clear_output(wait=True)
    print("Dataframe analyzed. 𓍊𓋼𓍊𓋼𓍊 𓃟")
    return df

def concentration(
    df=None, type=None, starttime=None, endtime=None, freq=None, flow=None
    ):

    """
    Compute particle concentration for a fixed interval or as a time series.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'count', 'Total', and 'Measured' columns and a datetime index.
    type : {"fixed", "series"}
        Whether to compute a single concentration or a time series.
    starttime, endtime : datetime-like
        Required if type="fixed".
    freq : string
        Resampling frequency (e.g., "1H") for type="series".
    flow : float
        Instrument flow rate in L/min.

    Returns
    -------
    float or pandas.Series
        Concentration(s) in particles per liter.
    """

    import pandas as pd
    import numpy as np

    ## --------------------------
    ## Basic input validation
    ## --------------------------
    if df is None or df.empty:
        return 0.
    
    if flow is None:
        raise ValueError("Flow rate (flow) must be provided.")

    if type not in ["fixed", "series"]:
        raise ValueError("type must be 'fixed' or 'series'.")

    ## Ensure required columns exist
    required = ["count", "Total", "Measured"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Input df must contain column '{col}'.")

    
    ## Safe copy
    df = df.copy()
    df[required] = df[required].fillna(0)

    ## --------------------------
    ## FIXED INTERVAL
    ## --------------------------

    if type == "fixed":

        if starttime is None or endtime is None:
            raise ValueError("starttime and endtime must be provided for type='fixed'.")

        ## Duration in minutes
        duration_min = (endtime - starttime) / pd.Timedelta(minutes=1)

        counts = df["count"].sum()

        ## Loss correction
        meas = df["Measured"]
        tot  = df["Total"]

        if meas.mean() == 0:
            loss_corr = 0.
        else:
            loss_corr = tot.mean() / meas.mean()

        ## Sampled volume (L)
        volume_factor = 0.165 * flow * duration_min

        conc = counts / volume_factor * loss_corr

        if not np.isfinite(conc):
            return 0.

        return conc

    ## --------------------------------------------------
    ## TIME SERIES
    ## --------------------------------------------------

    if type == "series":

        if freq is None:
            raise TypeError("A time frequency (freq) is required for type='series'.")

        ## Duration of each block in minutes
        block_minutes = pd.Timedelta(freq).total_seconds() / 60

        ## -----------------------------
        ## Count particles per block
        ## -----------------------------
        counts = df["count"].resample(freq).sum()

        ## -----------------------------
        ## Loss correction per block
        ## -----------------------------
        block_meas = df["Measured"].resample(freq).mean()
        block_tot  = df["Total"].resample(freq).mean()
        loss_corr = block_tot / block_meas
        loss_corr = loss_corr.replace([np.inf, -np.inf], np.nan).fillna(0)

        ## -----------------------------
        ## Sampled volume per block
        ## -----------------------------
        volume_factor = 0.165 * flow * block_minutes

        ## -----------------------------
        ## Final concentration series
        ## -----------------------------
        conc = (counts / volume_factor) * loss_corr
        conc = conc.replace([np.inf, -np.inf], np.nan).fillna(0)

        return conc

def preprocess_mbs(
    path_in,
    path_out,
    outfile,
    start,
    end=None,
    size_thresh=.8,
    verbose=True,
    classify=False,
    **classify_params
    ):
    
    """
    Preprocess raw MBS instrument CSV files into daily particle-level datasets.

    This function:
    - Loads all raw CSV files from a directory
    - Merges data for each day into a continuous time series
    - Includes rollover data: 
        - last part of the previous day (post-midnight segment)
        - excludes next-day data from the day's CSV but stores it for rollover
        - Next-day data is *not* included in the saved CSV, but is stored
          internally for the following day's preprocessing
    - Applies particle grouping, saturation detection, normalization,
      and fluorescence metrics
    - Applies class labels according to channel fluorescence
    - Writes one CSV per processed day (no returned DataFrame)

    Parameters
    ----------
    path_in : str
        Directory containing raw CSV files.
    path_out : str
        Output directory for processed daily CSVs.
    outfile : str
        Output file name prefix (without date) for processed daily CSVs.
    start : str or datetime-like
        Start date to process (YYYY-MM-DD).
    end : str or datetime-like or None, default=None
        End date. If None, only the start date is processed.
    size_thresh : float, default=.8
        Minimum particle size to include in output.
    verbose : bool, default=True
        Whether to print progress messages.
    """

    import pandas as pd
    import numpy as np
    import glob
    import os

    # ------------------------------------------------------
    # Helper function: read one raw file
    # ------------------------------------------------------

    def read_single_file(file, dum_row=33):

        """
        Reads one raw CSV data file and returns a DataFrame with a time index.
        """

        df1 = pd.read_csv(
            file, skiprows=dum_row,
            usecols=[
                'Time(ms)', 'FT', 'Size', 'Total', 'Measured',
                'AsymLR%', 'AsymLRinv%', 'PeakL', 'PeakR', 'MeanL', 'MeanR',
                'KurtosisL', 'KurtosisR', 'PeakMeanL', 'PeakMeanR',
                'PeakWidthL', 'PeakWidthR', 'PeakCountL', 'PeakCountR',
                'SumL', 'SumR', 'VarianceL', 'VarianceR',
                'MirrorL%', 'MirrorR%', 'SkewL', 'SkewR',
                'XE1_1', 'XE1_2', 'XE1_3', 'XE1_4', 'XE1_5', 'XE1_6',
                'XE1_7', 'XE1_8', 'TOF', 'Total', 'Measured'
            ]
        )

        ## Extract start time from header
        with open(file) as fp:
            for i, line in enumerate(fp):
                if i == dum_row - 1:
                    starttime = pd.to_datetime(
                        line[17:], format="%d/%m/%Y %H:%M:%S\n"
                    )
                    break

        ## Construct timeline
        df1["time"] = starttime + pd.to_timedelta(df1["Time(ms)"], unit="ms")
        df1 = df1.drop(columns=["Time(ms)"]).set_index("time")
        return df1

    ## ------------------------------------------------------
    ## Helper: Load all raw files for one date, including rollover
    ## ------------------------------------------------------

    def load_day_with_rollover(day, rollover):
        """
        Load all raw files for the given day, plus rollover if present.
        If rollover is NON-EMPTY, do NOT load previous day's last file.
        If rollover is EMPTY, load previous day's last file.
        """

        day_str = day.strftime("%Y%m%d")
        day_midnight = day.normalize()
        df = rollover.copy()
        today_files = sorted(glob.glob(os.path.join(path_in, f"*{day_str}*.csv")))

        ## If no rollover, include last file of previous day
        if rollover.empty:
            prev_day_str = (day - pd.Timedelta(days=1)).strftime("%Y%m%d")
            prev_files = sorted(glob.glob(os.path.join(path_in, f"*{prev_day_str}*.csv")))

            if prev_files:
                last_prev = prev_files[-1]
                df_prev = read_single_file(last_prev)
                df_prev = df_prev[df_prev.index >= day_midnight]
                df = pd.concat([df, df_prev])

        ## Load today's files
        for f in today_files:
            if verbose:
                print("Reading:", f)
            df_new = read_single_file(f)
            df = pd.concat([df, df_new])

        ## Cleaning up
        if not df.empty:
            df.index = pd.to_datetime(df.index)

            ## Remove any rows before today's midnight
            df = df[df.index >= day_midnight]

            ## Sort
            df = df.sort_index()

            ## Drop duplicate timestamps (keep earliest appearance)
            df = df[~df.index.duplicated(keep="first")]

        return df

    ## ------------------------------------------------------
    ## Date handling
    ## ------------------------------------------------------

    start = pd.to_datetime(start)
    if end is None:
        dates = [start]
    else:
        end = pd.to_datetime(end)
        dates = pd.date_range(start, end, freq="1D")
    rollover = pd.DataFrame(index=pd.to_datetime([]))

    ## ------------------------------------------------------
    ## MAIN LOOP
    ## ------------------------------------------------------

    for day in dates:

        ## Load data for this day, including rollover
        df = load_day_with_rollover(day, rollover)
        df = df[df.index >= day]

        ## Compute cutoff for next-day rollover (00:00 next day) and select
        next_midnight = (day + pd.Timedelta(days=1)).normalize()
        df_today = df[df.index < next_midnight].copy()
        rollover = df[df.index >= next_midnight].copy()
        if len(df_today) == 0:
            if verbose:
                print(f"No data found for {day.date()}. Skipping.")
            continue

        ## --------------------------------------------------
        ## Basic preprocessing
        ## --------------------------------------------------

        df_today["FL"] = df_today.iloc[:, :7].sum(axis=1)
        df_today.loc[df_today["FL"] < -550, :] = np.nan  ## clean bad values
        ft = df_today[df_today["FT"] == 1] # Thresholds from forced-trigger (FT==1)
        FL_th0 = ft.iloc[:, :8].std()
        FL_th1 = 3 * FL_th0
        FL_th2 = 9 * FL_th0

        ## --------------------------------------------------
        ## Particle selection
        ## --------------------------------------------------

        mat = df_today[(df_today["Size"] > size_thresh) &
                       (df_today["FT"] == 0)].copy()
        if len(mat) == 0:
            if verbose:
                print(f"No particles above threshold for {day.date()}.")
            continue
        mat = mat.apply(pd.to_numeric, errors="ignore") # ensure numeric

        ## --------------------------------------------------
        ## Group assignment (CP / FP / HFP / PBAP)
        ## --------------------------------------------------

        mat["group"] = "CP"
        cond_FP = (mat.iloc[:, :8] > FL_th1).any(axis=1)
        cond_HFP = (mat.iloc[:, :8] > FL_th2).any(axis=1)
        mat.loc[cond_FP, "group"] = "FP"
        mat.loc[cond_HFP, "group"] = "HFP"
        max_channel = mat.iloc[:, :8].idxmax(axis=1)
        pbap_mask = (mat["group"] == "HFP") & (max_channel == "XE1_2")
        mat.loc[pbap_mask, "group"] = "PBAP"

        ## --------------------------------------------------
        ## Label A–H based on threshold
        ## --------------------------------------------------

        mat["label"] = ""
        for i, ch in enumerate(mat.columns[:8]):
            mat["label"] += np.where(
                mat[ch] >= FL_th2.iloc[i], chr(65 + i), ""
            )

        ## --------------------------------------------------
        ## Saturation logic
        ## --------------------------------------------------

        FL_FTmean = ft.iloc[:, :8].mean()
        dum = mat.iloc[:, :8].add(FL_FTmean + FL_th1, axis=1)
        mat["saturated"] = np.where(dum.max(axis=1) >= 2048, "yes", "no")
        dum_sat = dum >= 2048
        mat["no_of_sat"] = dum_sat.sum(axis=1)

        ## Subgrouping
        mat["subgroup"] = "none"
        mat.loc[(mat["group"] == "HFP") & (mat["saturated"] == "no"),
                "subgroup"] = "HFP"
        mat.loc[(mat["group"] == "HFP") & (mat["saturated"] == "yes"),
                "subgroup"] = "HFPsat"
        mat.loc[(mat["group"] == "PBAP") & (mat["saturated"] == "no"),
                "subgroup"] = "PBAP"
        mat.loc[(mat["group"] == "PBAP") &
                (mat["saturated"] == "yes") &
                (mat["no_of_sat"] == 1),
                "subgroup"] = "PBAPsat"
        mat.loc[(mat["group"] == "PBAP") &
                (mat["saturated"] == "yes") &
                (mat["no_of_sat"] > 1),
                ["group", "subgroup"]] = ["HFP", "HFPsat"]

        ## --------------------------------------------------
        ## Fluorescence ratios
        ## --------------------------------------------------

        mat["FL_ratio"] = (
            mat.XE1_1 + mat.XE1_2 + mat.XE1_3
        ) / (mat.XE1_4 + mat.XE1_5 + mat.XE1_6 + mat.XE1_7 + mat.XE1_8)

        mat["FL_ratio_R"] = (
            mat.XE1_6 + mat.XE1_7 + mat.XE1_8
        ) / (
            mat.XE1_1 + mat.XE1_2 + mat.XE1_3 +
            mat.XE1_4 + mat.XE1_5
        )

        ## --------------------------------------------------
        ## Normalization
        ## --------------------------------------------------

        mat["min"] = mat.iloc[:, :8].min(axis=1)
        dum = mat.iloc[:, :8].subtract(mat["min"], axis=0)
        mat["max"] = dum.max(axis=1)
        mat_norm = dum.div(mat["max"], axis=0)
        mat_norm.columns = [f"{c}_norm" for c in mat_norm.columns]
        mat = pd.concat([mat, mat_norm], axis=1)

        ## --------------------------------------------------
        ## Optional particle identification in preprocessing
        ## --------------------------------------------------

        classify_params = classify_params or {}
        if classify:
            mat = pig(mat, clear=False, **classify_params)

        ## --------------------------------------------------
        ## Save daily output
        ## --------------------------------------------------

        outfile = os.path.join(path_out, f"%s_MBS_processed_{day:%Y%m%d}.csv" % outfile)
        mat.to_csv(outfile)
        if verbose:
            print("Saved:", outfile)
