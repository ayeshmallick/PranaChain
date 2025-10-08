# ckd_fetch_and_export.py
# ---------------------------------------------------------
# Combined script:
#  - Fixes SSL trust at runtime (uses certifi)
#  - Fetches UCI CKD dataset id=857
#  - Prints shapes, previews, metadata keys, variables preview
#  - Cleans awkward labels
#  - Exports easy-to-read CSVs:
#       * ckd_dataset_clean.csv
#       * ckd_target_summary.csv
#       * ckd_variables_info.csv
#       * ckd_metadata.csv
#       * ckd_value_counts/<one CSV per column>.csv
# ---------------------------------------------------------

import os
import re
import ssl
import unicodedata
import urllib.request
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Optional dependency (recommended). If missing, install with:
#   python -m pip install certifi
try:
    import certifi
except ImportError as e:
    raise SystemExit(
        "certifi is required for SSL fix. Install it with:\n"
        "  python -m pip install certifi"
    ) from e


# -----------------------------
# SSL fix: use certifi CA bundle
# -----------------------------
def configure_ssl_with_certifi() -> None:
    os.environ["SSL_CERT_FILE"] = certifi.where()
    ctx = ssl.create_default_context(cafile=certifi.where())
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    urllib.request.install_opener(opener)
    print("Using CA bundle:", certifi.where())


# -----------------------------
# Text cleaning helpers
# -----------------------------
def normalize_unicode(s: str):
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("≤", "<=").replace("≥", ">=").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s.strip())
    return s


def clean_albumin_label(v: str):
    if not isinstance(v, str):
        return v
    vv = normalize_unicode(v).lower()
    if vv == "1-jan":
        return "1"
    return normalize_unicode(v)


def clean_age_label(v: str):
    if not isinstance(v, str):
        return v
    vv = normalize_unicode(v)
    low = vv.lower()
    if low in {"< 12", "<12"}:
        return "0-12"
    if re.fullmatch(r"20-?dec", low):
        return "20-30"
    return vv


_range_rx_3 = re.compile(r"^\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)\s*$")
_range_rx_2 = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$")


def normalize_range_label(v: str):
    if not isinstance(v, str):
        return v
    s = normalize_unicode(v)
    m = _range_rx_3.match(s)
    if m:
        op, num = m.group(1), m.group(2)
        return f"{op}{num}"
    m = _range_rx_2.match(s)
    if m:
        a, b = m.group(1), m.group(2)
        try:
            if float(b) < float(a):
                a, b = b, a
        except Exception:
            pass
        return f"{a}-{b}"
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*([<>]=?)\s*", r"\1", s)
    return s


def clean_generic_label(v: str):
    if not isinstance(v, str):
        return v
    return normalize_range_label(normalize_unicode(v))


# -----------------------------
# Column-aware cleaning
# -----------------------------
def clean_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    if "age" in Xc.columns:
        Xc["age"] = Xc["age"].apply(clean_age_label)
    if "al" in Xc.columns:
        Xc["al"] = Xc["al"].apply(clean_albumin_label)
    for col in Xc.columns:
        if Xc[col].dtype == object:
            Xc[col] = Xc[col].apply(clean_generic_label)
    return Xc


# -----------------------------
# Save helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_value_counts(df: pd.DataFrame, outdir: str) -> None:
    ensure_dir(outdir)
    for col in df.columns:
        vc = df[col].value_counts(dropna=False).reset_index()
        vc.columns = [col, "count"]
        vc.to_csv(os.path.join(outdir, f"{col}_value_counts.csv"), index=False)


# -----------------------------
# Combined "test + export" flow
# -----------------------------
def main():
    configure_ssl_with_certifi()

    print("Fetching dataset id=857 (Risk Factor Prediction of Chronic Kidney Disease)...")
    ds = fetch_ucirepo(id=857)

    # Test-style prints (from your original ucirepo_test.py)
    X_raw = ds.data.features
    y = ds.data.targets

    print("Fetched dataset 857")
    print("Features shape:", X_raw.shape)
    print("Targets shape:", y.shape)

    print("\n--- First 5 feature rows ---")
    print(X_raw.head())

    print("\n--- Target summary ---")
    try:
        print(y.value_counts())
    except Exception:
        print(y.head())

    print("\n--- Metadata keys ---")
    print(list(ds.metadata.keys()))

    print("\n--- Variables (first 10) ---")
    print(ds.variables.head(10))

    # Clean and export (from export_ckd_tables.py)
    print("\nCleaning feature labels...")
    X = clean_dataframe(X_raw)
    data = pd.concat([X, y], axis=1)

    DATA_CSV = "ckd_dataset_clean.csv"
    TARGET_CSV = "ckd_target_summary.csv"
    VARS_CSV = "ckd_variables_info.csv"
    META_CSV = "ckd_metadata.csv"
    VC_DIR = "ckd_value_counts"

    print("Writing CSVs...")
    data.to_csv(DATA_CSV, index=False)

    target_summary = y.value_counts().reset_index()
    target_summary.columns = ["class", "count"]
    target_summary.to_csv(TARGET_CSV, index=False)

    ds.variables.to_csv(VARS_CSV, index=False)

    meta_df = pd.DataFrame(list(ds.metadata.items()), columns=["Key", "Value"])
    meta_df.to_csv(META_CSV, index=False)

    save_value_counts(data, VC_DIR)

    print("\nCSV files created:")
    print(f"- {DATA_CSV}")
    print(f"- {TARGET_CSV}")
    print(f"- {VARS_CSV}")
    print(f"- {META_CSV}")
    print(f"- {VC_DIR}/ (one CSV per column)")


if __name__ == "__main__":
    main()