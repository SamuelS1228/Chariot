
import streamlit as st
import pandas as pd
from optimization import optimize
from visualization import plot_network, summary

# ────────────────── page config ─────────────────────────────
st.set_page_config(page_title="Warehouse Optimizer", page_icon="🏭", layout="wide")

st.title("Warehouse Optimizer")
st.caption("Upload demand, define parameters, then run the solver.")

# init session state
if "scenario" not in st.session_state:
    st.session_state["scenario"] = {}

scn = st.session_state["scenario"]

with st.sidebar:
    st.header("Inputs")

    # Demand CSV
    up = st.file_uploader("Store Demand CSV (Longitude,Latitude,DemandLbs)", type=["csv"])
    if up:
        scn["demand_csv"] = up
        df_preview = pd.read_csv(up)
        st.write("Preview:")
        st.dataframe(df_preview.head())

    # Candidate warehouse CSV
    cand_up = st.file_uploader("Candidate Warehouse CSV (lon,lat)", type=["csv"])
    use_cand = st.checkbox("Restrict to candidate sites")
    if cand_up:
        scn["cand_csv"] = cand_up
        cand_df = pd.read_csv(cand_up, header=None)
        st.write("Loaded", len(cand_df), "candidate sites.")

    # Cost parameters
    rate_out = st.number_input("Outbound $/lb‑min", value=scn.get("rate_out", 0.02), key="rate_out")
    scn["rate_out"] = rate_out
    sqft_per_lb = st.number_input("Sq ft per lb", value=scn.get("sqft_per_lb", 0.02), key="sqft")
    cost_sqft = st.number_input("Variable $/sq ft / yr", value=scn.get("cost_sqft", 6.0), key="cost_sqft")
    fixed_cost = st.number_input("Fixed cost $/warehouse", value=scn.get("fixed_cost", 250000.0), key="fixed_cost", step=10000.0)

    # Drive time toggle
    use_drive = st.checkbox("Use real drive times (OpenRouteService)", value=scn.get("use_drive", False))
    scn["use_drive"] = use_drive
    ors_key = ""
    if use_drive:
        ors_key = st.text_input("ORS API key", type="password", key="ors")
    scn["ors_key"] = ors_key

    # k
    auto_k = st.checkbox("Optimize # warehouses", value=scn.get("auto_k", True))
    scn["auto_k"] = auto_k
    if auto_k:
        k_range = st.slider("k range", 1, 10, scn.get("k_rng", (2, 5)), key="k_rng")
        k_vals = list(range(k_range[0], k_range[1]+1))
    else:
        k_fixed = st.number_input("# warehouses", min_value=1, max_value=10, value=scn.get("k_fixed", 3), step=1, key="k_fixed")
        k_vals = [int(k_fixed)]

    scn["k_vals"] = k_vals

    # Inbound flow
    inbound_on = st.checkbox("Factor inbound flow", value=scn.get("inbound_on", False))
    scn["inbound_on"] = inbound_on
    inbound_pts = []
    inbound_rate = 0.0
    if inbound_on:
        inbound_rate = st.number_input("Inbound $/lb‑min", value=scn.get("in_rate", 0.01), key="in_rate")
        for i in range(5):
            with st.expander(f"Supply {i+1}", expanded=False):
                lon = st.number_input("Longitude", key=f"sup_lon_{i}", value=scn.get(f"sup_lon_{i}", 0.0), format="%.6f")
                lat = st.number_input("Latitude", key=f"sup_lat_{i}", value=scn.get(f"sup_lat_{i}", 0.0), format="%.6f")
                pct = st.number_input("% inbound flow", min_value=0.0, max_value=100.0, value=scn.get(f"sup_pct_{i}", 0.0), key=f"sup_pct_{i}")
                use = st.checkbox("Use", value=scn.get(f"sup_use_{i}", False), key=f"sup_use_{i}")
                scn[f"sup_lon_{i}"] = lon
                scn[f"sup_lat_{i}"] = lat
                scn[f"sup_pct_{i}"] = pct
                scn[f"sup_use_{i}"] = use
                if use and pct>0:
                    inbound_pts.append([lon, lat, pct/100])
    scn["inbound_pts"] = inbound_pts
    scn["in_rate"] = inbound_rate

    # RDC / SDC
    rdc_list = []
    for i in range(3):
        with st.expander(f"RDC/SDC {i+1}", expanded=False):
            enable = st.checkbox("Enable", key=f"rdc_en_{i}", value=scn.get(f"rdc_en_{i}", False))
            scn[f"rdc_en_{i}"] = enable
            if enable:
                lon = st.number_input("Longitude", key=f"rdc_lon_{i}", value=scn.get(f"rdc_lon_{i}", 0.0), format="%.6f")
                lat = st.number_input("Latitude", key=f"rdc_lat_{i}", value=scn.get(f"rdc_lat_{i}", 0.0), format="%.6f")
                typ = st.radio("Type", ["RDC (redistribute only)", "SDC (serve customers)"], key=f"rdc_type_{i}", index=0 if scn.get(f"rdc_type_{i}", "RDC").startswith("RDC") else 1)
                scn[f"rdc_lon_{i}"] = lon
                scn[f"rdc_lat_{i}"] = lat
                scn[f"rdc_type_{i}"] = typ
                rdc_list.append({"coords":[lon,lat], "is_sdc": typ.startswith("SDC")})
    scn["rdc_list"] = rdc_list
    trans_rate = st.number_input("Transfer $/lb‑min (RDC ➜ WH)", value=scn.get("trans_rate", 0.015), key="trans_rate")
    scn["trans_rate"] = trans_rate

    # Run solver
    if st.button("Run solver"):
        if "demand_csv" not in scn:
            st.error("Upload demand CSV first.")
            st.stop()
        df = pd.read_csv(scn["demand_csv"])
        # sanitize
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['DemandLbs'] = pd.to_numeric(df['DemandLbs'], errors='coerce')
        df = df.dropna(subset=['Longitude','Latitude','DemandLbs'])
        candidate_sites = None
        if use_cand and cand_up:
            cdf = pd.read_csv(cand_up, header=None)
            candidate_sites = cdf[[0,1]].dropna().values.tolist()
        res = optimize(
            df,
            k_vals,
            rate_out,
            sqft_per_lb,
            cost_sqft,
            fixed_cost,
            consider_inbound=inbound_on,
            inbound_rate_min=inbound_rate,
            inbound_pts=inbound_pts,
            fixed_centers=[],
            rdc_list=rdc_list,
            transfer_rate_min=trans_rate,
            use_drive_times=use_drive,
            ors_api_key=ors_key,
            candidate_centers=candidate_sites
        )
        scn["result"] = res
        st.success("Solver finished!")

# show results
if "result" in scn:
    r = scn["result"]
    plot_network(r["assigned"], r["centers"])
    summary(
        r["assigned"],
        r["total_cost"],
        r["out_cost"],
        r["in_cost"],
        r["trans_cost"],
        r["wh_cost"],
        r["centers"],
        r["demand_per_wh"],
        sqft_per_lb,
        rdc_enabled=any([not x['is_sdc'] for x in rdc_list]),
        consider_inbound=scn["inbound_on"],
        show_transfer=(r["trans_cost"]>0)
    )
