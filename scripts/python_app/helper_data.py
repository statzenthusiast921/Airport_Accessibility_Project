"""Airport graph data loading and lookups."""

import gc

import pandas as pd
import numpy as np

GITHUB_DATA_BASE = (
    "https://raw.githubusercontent.com/statzenthusiast921/"
    "Airport_Accessibility_Project/main/data/"
)

MASTER_AIR_PARQUET_URL = GITHUB_DATA_BASE + "master_air.parquet"
EDGES_AIRLINE_AIRPORT_MERGED_URL = GITHUB_DATA_BASE + "edges_airline_airport_merged.parquet"
EDGES_FEATURE_SIMILARITY_MERGED_URL = GITHUB_DATA_BASE + "edges_feature_similarity_merged.parquet"
EDGES_PROXIMITY_MERGED_URL = GITHUB_DATA_BASE + "edges_proximity_merged.parquet"
EDGES_SHARED_DESTINATIONS_MERGED_URL = (
    GITHUB_DATA_BASE + "edges_shared_destinations_merged.parquet"
)
EDGES_SHARED_DESTINATIONS_COSINE_MERGED_URL = (
    GITHUB_DATA_BASE + "edges_shared_destinations_cosine_merged.parquet"
)

CONN_AIRPORT_ALL = "__ALL__"

# Slim columns for gunicorn startup (dropdowns / labels only).
STARTUP_AIRPORT_COLUMNS = ["iata", "country", "display_name", "name"]

airport_df = None
_full_airport_loaded = False
carriers_edges = None
similarity_edges = None
proximity_edges = None
shared_destinations_edges = None
shared_destinations_cosine_edges = None

AIRLINE_IATA_TO_NAME = None
AIRPORT_IATA_META = None
SIMILARITY_AIRPORT_PROFILE = None
country_choices = None
airport_choices = None
country_airport_dict = None

PROXIMITY_EDGE_MAX_MILES = 250.0

SIMILARITY_Z_FEATURE_KEYS = (
    "z_connectivity_index",
    "z_log1p_num_dests",
    "z_redundancy_score",
    "z_elevation",
)

HOVER_COLS = [
    "latitude",
    "longitude",
    "display_name",
    "num_dests",
    "redundancy_score",
    "connectivity_index",
]

connection_type_choices = [
    "Carriers",
    "Statistical Similarity",
    "Proximity",
    "Shared Destinations",
    "Shared Destinations (Hub-Adjusted)",
]

CONNECTION_TYPE_DEFINITIONS = [
    (
        "Carriers",
        "Airlines linked to airports they fly to within the selected country. "
        "Useful for seeing which carriers cluster at which hubs (after the top-carrier filter).",
    ),
    (
        "Statistical Similarity",
        "Airports linked when connectivity index, redundancy score, destination count, "
        "and elevation are alike (z-scored, then compared). Link scores are 0–1 (higher = more similar).",
    ),
    (
        "Proximity",
        "Airports linked by short distance (within the proximity edge cap in the dataset). "
        "Shows geographic neighbors, not airline overlap.",
    ),
    (
        "Shared Destinations",
        "Airports linked by how many destination cities they both serve (same destination IATA in both route lists). "
        "The weight is the raw count — large hubs usually share more destinations simply because they fly everywhere.",
    ),
    (
        "Shared Destinations (Hub-Adjusted)",
        "Same underlying overlap as the raw count, but each link is scaled down when either airport serves "
        "a very large number of destinations (cosine-style adjustment). "
        "Makes overlap between a mega-hub and a mid-size airport easier to interpret fairly.",
    ),
]

_DEST_TABLE_COLUMNS = {
    "dest_name": "Destination",
    "connectivity_index": "Connectivity Index",
    "redundancy_score": "Redundancy Index",
}


def _column_values_are_hashable(series):
    sample = series.dropna().head(20)
    if sample.empty:
        return True
    for val in sample:
        if isinstance(val, (list, dict, np.ndarray)):
            return False
    return True


# Keep as plain strings — categorizing these breaks dropdown filters and Plotly maps.
_NO_CATEGORY_COLS = frozenset({
    "country",
    "city_name",
    "display_name",
    "iata",
    "name",
    "dest_name",
    "dest_iata",
})


def optimize_airport_df_memory(df):
    """Shrink string/numeric dtypes in place to lower RSS on small instances."""
    n = len(df)
    if n == 0:
        return df
    for col in _NO_CATEGORY_COLS:
        if col in df.columns and isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(str)
    for col in df.columns:
        if col in _NO_CATEGORY_COLS or col == "carriers":
            continue
        if df[col].dtype == object and _column_values_are_hashable(df[col]):
            try:
                nunique = df[col].nunique()
            except TypeError:
                continue
            if 0 < nunique < n * 0.5:
                df[col] = df[col].astype("category")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def load_startup_airport_index():
    """Load minimal airport columns for Render startup (avoids full master_air in RAM)."""
    global airport_df
    airport_df = pd.read_parquet(MASTER_AIR_PARQUET_URL, columns=STARTUP_AIRPORT_COLUMNS)
    for col in STARTUP_AIRPORT_COLUMNS:
        if col in airport_df.columns and airport_df[col].dtype == object:
            airport_df[col] = airport_df[col].astype(str)


def ensure_full_airport_df():
    """Load route-level master_air on first Airport / Airline Metrics use."""
    global airport_df, _full_airport_loaded
    if _full_airport_loaded:
        return airport_df
    full = pd.read_parquet(MASTER_AIR_PARQUET_URL)
    optimize_airport_df_memory(full)
    airport_df = full
    _full_airport_loaded = True
    gc.collect()
    return airport_df


def dest_table_for_airport(display_name):
    ensure_full_airport_df()
    mask = airport_df["display_name"] == display_name
    return airport_df.loc[
        mask,
        ["display_name", "dest_name", "dest_iata", "connectivity_index", "redundancy_score"],
    ].rename(columns=_DEST_TABLE_COLUMNS)


def build_airline_iata_to_name(airport_frame):
    mapping = {}
    for carriers in airport_frame["carriers"].dropna():
        if not isinstance(carriers, (list, tuple, np.ndarray)):
            continue
        for c in carriers:
            if not isinstance(c, dict):
                continue
            code = c.get("iata") or c.get("IATA") or c.get("airline_iata") or c.get("code")
            name = c.get("name")
            if code and name:
                code = str(code).strip().upper()
                if len(code) == 2:
                    mapping[code] = name
    return mapping


def build_similarity_airport_profiles():
    ap = airport_df.drop_duplicates(subset=["iata"]).copy()
    ap["num_dests"] = pd.to_numeric(ap["num_dests"], errors="coerce").fillna(0).astype(int)
    ap["connectivity_index"] = pd.to_numeric(ap["connectivity_index"], errors="coerce").fillna(0.0)
    ap["redundancy_score"] = pd.to_numeric(ap["redundancy_score"], errors="coerce").fillna(0.0)
    if "elevation" in ap.columns:
        ap["elevation"] = pd.to_numeric(ap["elevation"], errors="coerce").fillna(0.0)
    else:
        ap["elevation"] = 0.0
    ap["log1p_num_dests"] = np.log1p(ap["num_dests"].to_numpy(dtype=float))
    z_cols = ["connectivity_index", "log1p_num_dests", "redundancy_score", "elevation"]
    mat = ap[z_cols].to_numpy(dtype=float)
    mu = mat.mean(axis=0)
    sig = mat.std(axis=0)
    sig[sig == 0] = 1.0
    z_mat = (mat - mu) / sig
    profiles = {}
    for i, iata in enumerate(ap["iata"].astype(str)):
        profiles[iata] = {
            "connectivity_index": float(mat[i, 0]),
            "redundancy_score": float(mat[i, 2]),
            "num_dests": int(ap["num_dests"].iloc[i]),
            "elevation": float(mat[i, 3]),
            "z_connectivity_index": float(z_mat[i, 0]),
            "z_log1p_num_dests": float(z_mat[i, 1]),
            "z_redundancy_score": float(z_mat[i, 2]),
            "z_elevation": float(z_mat[i, 3]),
        }
    return profiles


def ensure_airline_iata_to_name():
    """Built on first Carriers network view (scans carriers column)."""
    global AIRLINE_IATA_TO_NAME
    if AIRLINE_IATA_TO_NAME is None:
        ensure_full_airport_df()
        AIRLINE_IATA_TO_NAME = build_airline_iata_to_name(airport_df)
    return AIRLINE_IATA_TO_NAME


def ensure_similarity_profiles():
    """Built on first Statistical Similarity view (heavy; not at app startup)."""
    global SIMILARITY_AIRPORT_PROFILE
    if SIMILARITY_AIRPORT_PROFILE is None:
        ensure_full_airport_df()
        SIMILARITY_AIRPORT_PROFILE = build_similarity_airport_profiles()
    return SIMILARITY_AIRPORT_PROFILE


def initialize_core_state():
    """Lightweight lookups for dropdowns and labels (safe at app startup)."""
    global AIRPORT_IATA_META
    global country_choices
    global airport_choices
    global country_airport_dict

    AIRPORT_IATA_META = (
        airport_df.drop_duplicates(subset=["iata"])
        .set_index("iata")[["display_name", "country"]]
        .to_dict("index")
    )
    country_choices = sorted(airport_df["country"].astype(str).unique())
    airport_choices = sorted(airport_df["display_name"].astype(str).unique())

    df_for_dict = airport_df[["country", "display_name"]]
    df_for_dict = df_for_dict.drop_duplicates(subset="display_name", keep="first")
    country_airport_dict = df_for_dict.groupby("country")["display_name"].apply(list).to_dict()


def _load_once(global_name, url):
    frame = globals()[global_name]
    if frame is None:
        globals()[global_name] = pd.read_parquet(url)
    return globals()[global_name]


def load_carriers_edges():
    return _load_once("carriers_edges", EDGES_AIRLINE_AIRPORT_MERGED_URL)


def load_similarity_edges():
    return _load_once("similarity_edges", EDGES_FEATURE_SIMILARITY_MERGED_URL)


def load_proximity_edges():
    return _load_once("proximity_edges", EDGES_PROXIMITY_MERGED_URL)


def load_shared_destinations_edges():
    return _load_once("shared_destinations_edges", EDGES_SHARED_DESTINATIONS_MERGED_URL)


def load_shared_destinations_cosine_edges():
    return _load_once(
        "shared_destinations_cosine_edges",
        EDGES_SHARED_DESTINATIONS_COSINE_MERGED_URL,
    )
