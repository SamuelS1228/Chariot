
import streamlit as st
import pandas as pd
from optimization import optimize
from visualization import plot_network, summary

st.set_page_config(page_title="Warehouse Network Optimizer", layout="wide")

st.title("📦 Warehouse & Distribution Network Optimizer")

# ─────────────────────────── Session bootstrap ────────────────────────────
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = {}

def _num_input(scn, key, label, default, fmt="%.4f", **kwargs):
    scn.setdefault(key, default)
    scn[key] = st.number_input(label, value=scn[key], format=fmt,
                               key=f"{key}_{scn['_name']}", **kwargs)

# ─────────────────────────── Sidebar renderer ─────────────────────────────
def render_sidebar(scn):
    name = scn["_name"]
    with st.sidebar:
        st.header(f"Inputs — {name}")

        # Demand upload ----------------------------------------------------
        up = st.file_uploader("Store demand CSV (Longitude, Latitude, DemandLbs)", key=f"dem_{name}")
        if up:
            scn["demand_file"] = up
        if "demand_file" not in scn:
            st.info("Upload a demand file to begin.")
            st.stop()

        if st.expander("Preview demand file").checkbox("Show head", key=f"pr_{name}"):
            st.dataframe(pd.read_csv(scn["demand_file"]).head())

        # Cost parameters --------------------------------------------------
        st.subheader("Cost Parameters ($ / lb‑minute)")
        _num_input(scn, "rate_out", "Outbound", 0.02, "%.4f")
        _num_input(scn, "in_rate", "Inbound", 0.01, "%.4f")
        _num_input(scn, "trans_rate", "Transfer (RDC ➜ WH)", 0.015, "%.4f")

        st.subheader("Warehouse Cost Parameters")
        _num_input(scn, "sqft_per_lb", "Sq ft per lb", 0.02, "%.4f")
        _num_input(scn, "cost_sqft", "Variable $/sq ft / yr", 6.0, "%.2f")
        _num_input(scn, "fixed_wh_cost", "Fixed $/warehouse", 250000.0, "%.0f", step=50000.0)

        # Drive‑time toggle -----------------------------------------------
        scn["drive_times"] = st.checkbox("Use real drive times (OpenRouteService)",
                                         value=scn.get("drive_times", False),
                                         key=f"dt_{name}")
        if scn["drive_times"]:
            scn["ors_key"] = st.text_input("OpenRouteService API key",
                                           value=scn.get("ors_key", ""),
                                           type="password",
                                           key=f"ors_{name}")

        # k selection ------------------------------------------------------
        st.subheader("# Warehouses")
        scn["auto_k"] = st.checkbox("Optimize # warehouses",
                                    value=scn.get("auto_k", True),
                                    key=f"ak_{name}")
        if scn["auto_k"]:
            scn["k_rng"] = st.slider("k range", 1, 10, scn.get("k_rng", (2, 5)),
                                     key=f"k_rng_{name}")
            k_vals = list(range(int(scn["k_rng"][0]),
                                int(scn["k_rng"][1]) + 1))
        else:
            _num_input(scn, "k_fixed", "# warehouses", 3, "%.0f",
                       step=1, min_value=1, max_value=10)
            k_vals = [int(scn["k_fixed"])]

        # Fixed warehouses -------------------------------------------------
        st.subheader("Fixed Warehouses")
        fixed_text = st.text_area("lon,lat per line",
                                  value=scn.get("fixed_text", ""),
                                  key=f"fx_{name}",
                                  height=80)
        scn["fixed_text"] = fixed_text
        fixed_centers = []
        for ln in fixed_text.splitlines():
            try:
                lon, lat = map(float, ln.split(","))
                fixed_centers.append([lon, lat])
            except Exception:
                continue
        scn["fixed_centers"] = fixed_centers

        # Candidate warehouse CSV -----------------------------------------
        st.subheader("Candidate Warehouse Sites")
        cand_up = st.file_uploader("Candidate CSV (lon,lat)", key=f"cand_{name}")
        if cand_up:
            scn["cand_file"] = cand_up
        scn["restrict_cand"] = st.checkbox("Restrict to candidate sites only",
                                           value=scn.get("restrict_cand", False),
                                           key=f"rc_{name}")

        # Inbound flow -----------------------------------------------------
        st.subheader("Inbound Flow (supply points)")
        scn["inbound_on"] = st.checkbox("Include inbound flow",
                                        value=scn.get("inbound_on", False),
                                        key=f"inb_{name}")
        inbound_pts = []
        if scn["inbound_on"]:
            sup_text = st.text_area("lon,lat,pct (pct 0-100) per line",
                                    value=scn.get("sup_text", ""),
                                    key=f"sup_{name}",
                                    height=100)
            scn["sup_text"] = sup_text
            for ln in sup_text.splitlines():
                try:
                    lon, lat, pct = map(float, ln.split(","))
                    inbound_pts.append([lon, lat, pct / 100.0])
                except Exception:
                    continue
        scn["inbound_pts"] = inbound_pts

        # RDC / SDC --------------------------------------------------------
        st.subheader("RDC / SDC (up to 3)")
        rdc_list = []
        for idx in range(1, 4):
            exp = st.expander(f"Center {idx}")
            with exp:
                en = st.checkbox("Enable", key=f"rdc_en_{name}_{idx}",
                                 value=scn.get(f"rdc{idx}_en", False))
                scn[f"rdc{idx}_en"] = en
                if en:
                    lon = st.number_input("Longitude", format="%.6f",
                                          value=float(scn.get(f"rdc{idx}_lon", 0.0)),
                                          key=f"rdc_lon_{name}_{idx}")
                    lat = st.number_input("Latitude", format="%.6f",
                                          value=float(scn.get(f"rdc{idx}_lat", 0.0)),
                                          key=f"rdc_lat_{name}_{idx}")
                    typ = st.radio("Type", ["RDC", "SDC"],
                                   index=0 if scn.get(f"rdc{idx}_typ", "RDC") == "RDC" else 1,
                                   key=f"rdc_typ_{name}_{idx}", horizontal=True)
                    scn[f"rdc{idx}_lon"] = lon
                    scn[f"rdc{idx}_lat"] = lat
                    scn[f"rdc{idx}_typ"] = typ
                    rdc_list.append({"coords": [lon, lat], "is_sdc": typ == "SDC"})
        scn["rdc_list"] = rdc_list
        _num_input(scn, "rdc_sqft_per_lb", "RDC Sq ft per lb", scn.get("sqft_per_lb", 0.02))
        _num_input(scn, "rdc_cost_sqft", "RDC $/sq ft / yr", scn.get("cost_sqft", 6.0), "%.2f")

        # Run button -------------------------------------------------------
        if st.button("Run solver", key=f"run_{name}"):
            st.session_state["run_target"] = name
            st.session_state["k_vals"] = k_vals

# ─────────────────────────── Main layout ────────────────────────────────
tab_names = list(st.session_state["scenarios"].keys()) + ["➕ New scenario"]
tabs = st.tabs(tab_names)

# Existing scenarios ------------------------------------------------------
for i, t in enumerate(tabs[:-1]):
    name = tab_names[i]
    scn = st.session_state["scenarios"][name]
    scn["_name"] = name
    with t:
        render_sidebar(scn)
        if st.session_state.get("run_target") == name:
            with st.spinner("Running optimization…"):
                # Demand data
                df = pd.read_csv(scn["demand_file"])
                for col in ["Longitude", "Latitude", "DemandLbs"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["Longitude", "Latitude", "DemandLbs"])
                # Candidate sites
                candidate_sites = None
                if scn.get("cand_file") and scn.get("restrict_cand"):
                    cand_df = pd.read_csv(scn["cand_file"], header=None)
                    cand_df = cand_df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
                    candidate_sites = cand_df.values.tolist()

                res = optimize(
                    df,
                    st.session_state["k_vals"],
                    scn["rate_out"],
                    scn["sqft_per_lb"],
                    scn["cost_sqft"],
                    scn["fixed_wh_cost"],
                    consider_inbound=scn["inbound_on"],
                    inbound_rate_min=scn["in_rate"],
                    inbound_pts=scn["inbound_pts"],
                    fixed_centers=scn["fixed_centers"],
                    rdc_list=scn["rdc_list"],
                    transfer_rate_min=scn["trans_rate"],
                    rdc_sqft_per_lb=scn["rdc_sqft_per_lb"],
                    rdc_cost_per_sqft=scn["rdc_cost_sqft"],
                    use_drive_times=scn["drive_times"],
                    ors_api_key=scn.get("ors_key", ""),
                    candidate_sites=candidate_sites,
                )
            plot_network(res["assigned"], res["centers"])
            summary(
                res["assigned"], res["total_cost"], res["out_cost"], res["in_cost"],
                res["trans_cost"], res["wh_cost"], res["centers"], res["demand_per_wh"],
                scn["sqft_per_lb"], rdc_enabled=len(res.get("rdc_only_idx", [])) > 0,
                consider_inbound=scn["inbound_on"], show_transfer=res["trans_cost"] > 0,
            )

# New scenario tab --------------------------------------------------------
with tabs[-1]:
    new_name = st.text_input("Scenario name")
    if st.button("Create scenario"):
        if new_name and new_name not in st.session_state["scenarios"]:
            st.session_state["scenarios"][new_name] = {}
            st.experimental_rerun()
