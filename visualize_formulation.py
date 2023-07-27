"""visualize_formulation.py
Creates figures of molecular weight and radius against temperature from an input file

Input file has columns, 
    "Well",
    "DLS Temp (C)",
    "MW-S (Da)",
    "Radius (nm)",
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


THRESHOLD_TEMPERATURE = 50
WELLS_WITH_DATA = [
    ["C1", "C3"],
    ["D1", "D3"],
    ["E1", "E3"],
    ["F1", "F3"],
]
# WELL_TO_FORM is used to map the well ID to the formulation if the "formulation" column is not
# available in CSV
WELL_TO_FORM = {
    "C1": "Acetate pH 5.0",
    "C3": "Histidine pH 7.0",
    "D1": "Citrate pH 6.2",
    "D3": "Histidine pH 5.5",
    "E1": "Acetate pH 4.5",
    "E3": "Histidine pH 6.2",
    "F1": "Histidine pH 5.5",
    "F3": "Acetate pH 5.0",
}
FILE_PATH = ""
PATH_SAVE = "" # path to which to save figs
# RENAME_COLS should not be changed
RENAME_COLS = {
    "mw_s_da": "metric_value-mw_s_da",
    "radius_nm": "metric_value-radius_nm"
}
logging.basicConfig(filename="visualize_formulation.log", level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_data(df_data):
    """Cleans dataframe for nulls, odd characters, and enforces types

    Parameters
    ----------
    df_data : pd.DataFrame
        Contains "data_quality", "dls_temp_c", "mw_s_da", and "radius_nm"

    Returns
    -------
    pd.DataFrame
        Contains "data_quality", "dls_temp_c", "mw_s_da", and "radius_nm"
    """
    df_data.columns = (
        df_data.columns.str.lower()
        .str.replace("-", "_")
        .str.replace(" ", "_")
        .str.replace("(", "", regex=True)
        .str.replace(")", "", regex=True)
    )
    # keep cols w relevant data
    flattened = [val for sublist in WELLS_WITH_DATA for val in sublist]
    df_data = df_data.query("well in @flattened").copy()
    # drop data where data missing for all columns
    df_data = df_data.query("data_quality != 'Error'").copy()
    df_data = df_data[~df_data.data_quality.isna()].copy()
    # replace '--' with null
    for col in ["dls_temp_c", "mw_s_da", "radius_nm"]:
        if df_data[col].dtype != np.float64:
            df_data.loc[df_data[col].str.contains("--"), col] = np.nan
    return df_data.assign(
        dls_temp_c=lambda x: x.dls_temp_c.astype(np.float64),
        mw_s_da=lambda x: x.mw_s_da.astype(np.float64),
        radius_nm=lambda x: x.radius_nm.astype(np.float64),
    )


def remove_spurious_points(df_data):
    """Identify spurious points to null for "radius_nm"

    Parameters
    ----------
    df_data : pd.DataFrame
        Contains "dls_temp_c" and "radius_nm"

    Returns
    -------
    pd.DataFrame
        Contains same columns as input
    """
    df_grp_list = []
    for well, df_grp in df_data.groupby("well"):
        are_spurious_points = (
            df_grp.dls_temp_c < THRESHOLD_TEMPERATURE)
            & (df_grp.radius_nm > df_grp.radius_nm.mean()
        )
        if are_spurious_points.any():
            print(f"Dropping {sum(are_spurious_points)} spurious points for {well}")
            df_grp.loc[are_spurious_points, "radius_nm"] = np.nan
        df_grp_list.append(df_grp)
    return pd.concat(df_grp_list, ignore_index=True)


def prepare_data(file_path=FILE_PATH):
    """Reformats the dataframe for 

    Parameters
    ----------
    file_path : str, optional
        Path of the file location, by default FILE_PATH

    Returns
    -------
    pd.DataFrame
        Dataframe with columns "well", "dls temp (c)", "formulation", "metric_name",
        "metric_value", and "measurement"
    """
    df_data = (
        pd.read_csv(file_path)
        .pipe(clean_data)
        .pipe(remove_spurious_points)
        .rename(columns=RENAME_COLS)
    )
    df_data = pd.wide_to_long(
        df_data,
        stubnames="metric_value",
        i=["well", "dls_temp_c", "measurement"],
        j="metric_name",
        sep="-",
        suffix=r"\D+",
    ).reset_index()
    if "formulation" not in df_data:
        logger.info("Mapping well to formulation with %s", WELL_TO_FORM)
        df_data= df_data.assign(formulation=lambda x: x.well.map(WELL_TO_FORM))
    return df_data


def config_legend_contents(ax, ax2):
    """Helper function to combine legend labels from two axes

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes for left-hand-side
    ax2 : matplotlib.axes.Axes
        Second axes for right-hand-side

    Returns
    -------
    tuple[list, list]
        [0] Contains matplotlib matplotlib.lines.Line2D objects
        [1] Contains list of str labels
    """
    handles, labels = ax.get_legend_handles_labels()
    second_handles, second_labels = ax2.get_legend_handles_labels()
    new_handles = [h for h in handles]
    new_labels = [l for l in labels]
    new_handles.append(second_handles[-1])
    new_labels.append(second_labels[-1])
    return new_handles, new_labels


def fig_rad_mw_temp(df_wells, path_save=PATH_SAVE):
    """Generate a figure of radius and molecular weight against temperature

    Parameters
    ----------
    df_wells : pd.DataFrame
        Contains "dls temp (c)", "formulation", "metric_name", and "metric_value" columns 
    path_save : str, optional
        Path to which to save figures, by default PATH_SAVE

    Returns
    -------
    tuple[matplotlib.axes.Axes]
        Objects pointing to figure
    """
    fig, ax = plt.subplots(1, 1, dpi=150)
    sns.lineplot(
        data=df_wells.query("metric_name == 'mw_s_da'"),
        x="dls_temp_c",
        y="metric_value",
        hue="formulation",
        style="metric_name",
        markers=True,
        markersize=12,
        mec="k",
        ax=ax,
    ).set_ylabel("MW-S (Da)")
    ax2 = plt.twinx()
    sns.lineplot(
        data=df_wells.query("metric_name == 'radius_nm'"),
        x="dls_temp_c",
        y="metric_value",
        hue="formulation",
        style="metric_name",
        ax=ax2, 
        markers=["^"],
        markersize=12,
        mec="k",
        legend="brief",
    ).set_ylabel("Radius (nm)")
    ax.set_xlabel("Temperature ($\degree$C)")
    ax.legend(*config_legend_contents(ax, ax2))
    ax2.get_legend().remove()
    plt.savefig(f"{'_'.join(df_wells.well.unique())}.png")
    return fig, ax, ax2


def make_figs():
    logger.info("Preparing data...")
    df_data = prepare_data()
    for wells in WELLS_WITH_DATA:
        df_wells = df_data.query("well in @wells")
        fig_rad_mw_temp(df_wells)
        logger.info("Successfuly made figure for wells %s", wells)


def test_figs():
    df_wells = prepare_data("dummy_protein.csv")
    return fig_rad_mw_temp(df_wells)


if __name__ == "__main__":
    make_figs()
