
import streamlit as st
import pandas as pd
from optimization import optimize
from visualization import plot_network, summary

st.set_page_config(
    page_title='Warehouse Optimizer – Scenarios',
    page_icon='🏭',
    layout='wide',
)
st.title('Warehouse Optimizer — Scenario Workspace')
st.caption(
    'Create multiple scenarios, including optional two‑echelon networks (up to 3 RDC/SDC), '
    'then run the solver and export results.'
)

# ────────────────── bootstrap session state ───────────────────────
if 'scenarios' not in st.session_state:
    st.session_state['scenarios'] = {}  # name ➞ dict

# ────────────────── tabs (one per scenario + “new”) ───────────────
scenario_names = list(st.session_state['scenarios'])
tabs = scenario_names + ['➕  New scenario']
tab_refs = st.tabs(tabs)

# ────────────────── “New scenario” tab ────────────────────────────
with tab_refs[-1]:
    new_name = st.text_input('Scenario name')
    if st.button('Create') and new_name:
        if new_name in st.session_state['scenarios']:
            st.warning('That name already exists.')
        else:
            st.session_state['scenarios'][new_name] = {}
            st.success('Scenario created.')
            st.experimental_rerun()

# ────────────────── iterate existing scenarios ────────────────────
for idx, name in enumerate(scenario_names):
    scenario = st.session_state['scenarios'][name]

    with tab_refs[idx]:
        st.header(f'Scenario: {name}')

        # ── File upload ───────────────────────────────────────────
        up = st.file_uploader('Store demand CSV', key=f'up_{name}')
        if up:
            scenario['upload'] = up
        if 'upload' in scenario and st.checkbox(
            'Show preview', key=f'prev_{name}'
        ):
            st.dataframe(pd.read_csv(scenario['upload']).head())

        # ── COST PARAMETERS ──────────────────────────────────────
        def n(key, label, default, fmt='%.10f', **k):
            scenario.setdefault(key, default)
            scenario[key] = st.number_input(
                label,
                value=scenario[key],
                format=fmt,
                key=f'{name}_{key}',
                **k,
            )

        n('rate_out', 'Outbound $/lb‑mi', 0.02)
        n('fixed_cost', 'Fixed cost $/warehouse', 250000.0, '%.2f', step=50000.0)
        n('sqft_per_lb', 'Sq ft per lb', 0.02)
        n('cost_sqft', 'Variable $/sq ft / yr', 6.0, '%.2f')

        scenario.setdefault('auto_k', True)
        scenario['auto_k'] = st.checkbox(
            'Optimize # warehouses', value=scenario['auto_k'], key=f'auto_{name}'
        )
        if scenario['auto_k']:
            scenario.setdefault('k_rng', (2, 5))
            scenario['k_rng'] = st.slider(
                'k range', 1, 10, scenario['k_rng'], key=f'k_rng_{name}'
            )
            k_vals_ui = range(
                int(scenario['k_rng'][0]), int(scenario['k_rng'][1]) + 1
            )
        else:
            n(
                'k_fixed',
                '# warehouses',
                3,
                '%.0f',
                step=1,
                min_value=1,
                max_value=10,
            )
            k_vals_ui = [int(scenario['k_fixed'])]

        # ── FIXED WAREHOUSES ─────────────────────────────────────
        st.subheader('Fixed Warehouses (up to 10)')
        scenario.setdefault('fixed', [[0.0, 0.0, False] for _ in range(10)])
        for i in range(10):
            with st.expander(f'Fixed Warehouse {i+1}', expanded=False):
                lat = st.number_input(
                    'Latitude',
                    value=scenario['fixed'][i][1],
                    key=f'{name}_fw_lat{i}',
                    format='%.6f',
                )
                lon = st.number_input(
                    'Longitude',
                    value=scenario['fixed'][i][0],
                    key=f'{name}_fw_lon{i}',
                    format='%.6f',
                )
                use = st.checkbox(
                    'Use this location',
                    value=scenario['fixed'][i][2],
                    key=f'{name}_fw_use{i}',
                )
                scenario['fixed'][i] = [lon, lat, use]
        fixed_centers = [
            [lon, lat] for lon, lat, use in scenario['fixed'] if use
        ]

        # ── INBOUND SUPPLY (direct to WH OR via RDC) ─────────────
        scenario.setdefault('inbound_on', False)
        scenario['inbound_on'] = st.checkbox(
            'Factor inbound flow', value=scenario['inbound_on'], key=f'in_on_{name}'
        )
        inbound_rate = 0.0
        inbound_pts = []
        if scenario['inbound_on']:
            n('in_rate', 'Inbound $/lb‑mi', 0.01)
            inbound_rate = scenario['in_rate']
            scenario.setdefault('sup', [[0.0, 0.0, 0.0, False] for _ in range(5)])
            for j in range(5):
                with st.expander(f'Supply Point {j+1}', expanded=False):
                    slat = st.number_input(
                        'Latitude',
                        value=scenario['sup'][j][1],
                        key=f'{name}_sp_lat{j}',
                        format='%.6f',
                    )
                    slon = st.number_input(
                        'Longitude',
                        value=scenario['sup'][j][0],
                        key=f'{name}_sp_lon{j}',
                        format='%.6f',
                    )
                    pct = st.number_input(
                        '% inbound flow',
                        min_value=0.0,
                        max_value=100.0,
                        value=scenario['sup'][j][2],
                        key=f'{name}_sp_pct{j}',
                        format='%.2f',
                    )
                    use_sp = st.checkbox(
                        'Use this supply point',
                        value=scenario['sup'][j][3],
                        key=f'{name}_sp_use{j}',
                    )
                    scenario['sup'][j] = [slon, slat, pct, use_sp]
            inbound_pts = [
                [lon, lat, pct / 100]
                for lon, lat, pct, use in scenario['sup']
                if use and pct > 0
            ]

        # ── MULTIPLE RDC / SDC ─────────────────────────────────
        st.subheader('Redistribution / Service Distribution Centers (up to 3)')
        scenario.setdefault(
            'rdcs',
            [
                {'enabled': False, 'lon': 0.0, 'lat': 0.0, 'type': 'RDC'}
                for _ in range(3)
            ],
        )
        rdc_list = []
        for i in range(3):
            rd = scenario['rdcs'][i]
            with st.expander(f'Center {i+1}', expanded=False):
                rd['enabled'] = st.checkbox(
                    'Enable', value=rd['enabled'], key=f'{name}_rdc_enable{i}'
                )
                if rd['enabled']:
                    rd['lat'] = st.number_input(
                        'Latitude',
                        value=rd['lat'],
                        key=f'{name}_rdc_lat{i}',
                        format='%.6f',
                    )
                    rd['lon'] = st.number_input(
                        'Longitude',
                        value=rd['lon'],
                        key=f'{name}_rdc_lon{i}',
                        format='%.6f',
                    )
                    rd['type'] = st.radio(
                        'Center type',
                        ['RDC (redistribute only)', 'SDC (redistribute + serve customers)'],
                        index=0 if rd['type'] == 'RDC' else 1,
                        key=f'{name}_rdc_type{i}',
                    )
                scenario['rdcs'][i] = rd
                if rd['enabled']:
                    rdc_list.append(
                        {
                            'coords': [rd['lon'], rd['lat']],
                            'is_sdc': rd['type'].startswith('SDC'),
                        }
                    )

        # ── TRANSFER & RDC COST PARAMETERS ────────────────────
        n('trans_rate', 'Transfer $/lb‑mi (RDC ➜ WH)', 0.015)
        transfer_rate = scenario['trans_rate']
        n('rdc_sqft_per_lb', 'RDC Sq ft per lb shipped', scenario.get('sqft_per_lb', 0.02))
        n('rdc_cost_sqft', 'RDC variable $/sq ft / yr', scenario.get('cost_sqft', 6.0), '%.2f')

        # ── RUN SOLVER BUTTON ───────────────────────────────────
        if st.button('Run solver', key=f'run_{name}'):
            if 'upload' not in scenario:
                st.warning('Upload a CSV first.')
            elif scenario['inbound_on'] and not inbound_pts:
                st.warning('Enable at least one supply point.')
            else:
                df = pd.read_csv(scenario['upload'])
                result = optimize(df,
                    k_vals_ui,
                    scenario['rate_out'],
                    scenario['sqft_per_lb'],
                    scenario['cost_sqft'],
                    scenario['fixed_cost'],
                    consider_inbound=scenario['inbound_on'],
                    inbound_rate=inbound_rate,
                    inbound_pts=inbound_pts,
                    fixed_centers=fixed_centers,
                    rdc_list=rdc_list,
                    transfer_rate=transfer_rate,
                    rdc_sqft_per_lb=scenario.get('rdc_sqft_per_lb', restrict_candidates=scenario.get('restrict_candidates', False), candidate_centers=scenario.get('candidate_centers', [])),
                    rdc_cost_per_sqft=scenario.get('rdc_cost_sqft'),
                )
                scenario['result'] = result
                st.success('Solver finished.')

        # ── RESULTS VISUALISATION ───────────────────────────────
        if 'result' in scenario:
            r = scenario['result']
            plot_network(r['assigned'], r['centers'])
            summary(
                r['assigned'],
                r['total_cost'],
                r['out_cost'],
                r['in_cost'],
                r['trans_cost'],
                r['wh_cost'],
                r['centers'],
                r['demand_per_wh'],
                scenario['sqft_per_lb'],
                rdc_enabled=len(r.get('rdc_only_idx', [])) > 0,
                rdc_idx=None,
                rdc_sqft_per_lb=scenario.get('rdc_sqft_per_lb'),
                consider_inbound=scenario['inbound_on'],
                show_transfer=(len(r.get('rdc_only_idx', [])) > 0 and r['trans_cost'] > 0),
            )

            csv = r['assigned'].to_csv(index=False).encode()
            st.download_button(
                'Download assignment CSV',
                csv,
                file_name=f'{name}_assignment.csv',
                key=f'dl_{name}',
            )

        # ── candidate warehouses ─────────────────────────────
        st.subheader("Candidate Warehouse Locations")
        cand_up = st.file_uploader("Candidate sites CSV (lon,lat)", key=f"cand_{name}")
        if cand_up:
            # try no header, then with header
            try:
                cand_df = pd.read_csv(cand_up, header=None)
            except Exception:
                cand_df = pd.read_csv(cand_up)
            cand_df = cand_df.iloc[:, :2]
            cand_df.columns = ['Longitude','Latitude']
            cand_df['Longitude'] = pd.to_numeric(cand_df['Longitude'], errors='coerce')
            cand_df['Latitude'] = pd.to_numeric(cand_df['Latitude'], errors='coerce')
            cand_df = cand_df.dropna()
            scenario['candidate_centers'] = cand_df[['Longitude','Latitude']].values.tolist()
        if 'candidate_centers' in scenario:
            st.dataframe(pd.DataFrame(scenario['candidate_centers'], columns=['Lon','Lat']).head())
        scenario.setdefault('restrict_candidates', False)
        scenario['restrict_candidates'] = st.checkbox(
            "Restrict to these candidate sites", value=scenario['restrict_candidates'], key=f"cand_only_{name}")
