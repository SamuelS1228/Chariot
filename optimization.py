
import numpy as np
from sklearn.cluster import KMeans
from utils import warehousing_cost

EARTH_RADIUS_MILES = 3958.8

def _haversine_vec(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return EARTH_RADIUS_MILES * 2 * np.arcsin(np.sqrt(a))

# ─────────────────────────────────────────────────────────────
# helpers for transfers & inbound
# ─────────────────────────────────────────────────────────────
def _transfer_cost(rdc_coord, other_centers, demand_per_wh, transfer_rate):
    if transfer_rate == 0 or not other_centers:
        return 0.0
    r_lon, r_lat = rdc_coord
    oc = np.asarray(other_centers)
    dist = _haversine_vec(
        np.full(len(oc), r_lon),
        np.full(len(oc), r_lat),
        oc[:, 0],
        oc[:, 1],
    )
    demand = np.asarray(demand_per_wh, dtype=float)
    return float((dist * demand * transfer_rate).sum())

def _transfer_cost_multi_rdcs(rdc_only_coords, centers, rdc_only_idx, demand_per_wh, transfer_rate):
    total = 0.0
    for coord, idx in zip(rdc_only_coords, rdc_only_idx):
        other_idx = [i for i in range(len(centers)) if i not in rdc_only_idx and i != idx]
        other_centers = [centers[i] for i in other_idx]
        other_demand = [demand_per_wh[i] for i in other_idx]
        total += _transfer_cost(coord, other_centers, other_demand, transfer_rate)
    return total

def _transfer_cost_multi(inbound_pts, centers, demand_per_wh, inbound_rate):
    if not inbound_pts:
        return 0.0
    centers = np.asarray(centers)
    c_lon = centers[:, 0]
    c_lat = centers[:, 1]
    demand_per_wh = np.asarray(demand_per_wh)
    cost = 0.0
    for lon, lat, pct in inbound_pts:
        dist = _haversine_vec(np.full_like(c_lon, lon), np.full_like(c_lat, lat), c_lon, c_lat)
        cost += (dist * demand_per_wh * pct * inbound_rate).sum()
    return cost

def _inbound_cost_to_multiple_rdcs(total_demand, inbound_pts, inbound_rate, rdc_only_coords):
    if not inbound_pts or not rdc_only_coords:
        return 0.0
    share = total_demand / len(rdc_only_coords)
    cost = 0.0
    for r_lon, r_lat in rdc_only_coords:
        for lon, lat, pct in inbound_pts:
            dist = _haversine_vec(r_lon, r_lat, lon, lat)
            cost += dist * share * pct * inbound_rate
    return cost

def _assign(df, centers, forbid_idx=None):
    s_lat = df['Latitude'].values
    s_lon = df['Longitude'].values
    dists = np.empty((len(df), len(centers)))
    for j, (clon, clat) in enumerate(centers):
        dists[:, j] = _haversine_vec(s_lon, s_lat, clon, clat)
    if forbid_idx is not None:
        dists[:, forbid_idx] = 1e12
    idx = dists.argmin(axis=1)
    return idx, dists[np.arange(len(df)), idx]

# ─────────────────────────────────────────────────────────────
# evaluate network
# ─────────────────────────────────────────────────────────────
def evaluate(
    df,
    centers,
    rate_out,
    sqft_per_lb,
    cost_per_sqft,
    fixed_cost,
    consider_inbound=False,
    inbound_rate=0.0,
    inbound_pts=None,
    rdc_only_idx=None,
    rdc_only_coords=None,
    transfer_rate=0.0,
    rdc_sqft_per_lb=None,
    rdc_cost_per_sqft=None,
):
    if rdc_only_idx is None:
        rdc_only_idx = []
    if rdc_only_coords is None:
        rdc_only_coords = []

    forbid_idx = rdc_only_idx if rdc_only_idx else None

    idx, dist = _assign(df, centers, forbid_idx=forbid_idx)
    assigned = df.copy()
    assigned['Warehouse'] = idx
    assigned['DistMiles'] = dist

    # outbound cost
    outbound_cost = (assigned['DistMiles'] * assigned['DemandLbs'] * rate_out).sum()

    # warehousing cost
    wh_cost = 0.0
    demand_list = []
    for i in range(len(centers)):
        demand = assigned.loc[assigned['Warehouse'] == i, 'DemandLbs'].sum()
        demand_list.append(demand)
        if i in rdc_only_idx and (rdc_sqft_per_lb is not None) and (rdc_cost_per_sqft is not None):
            _sq = rdc_sqft_per_lb
            _csq = rdc_cost_per_sqft
        else:
            _sq = sqft_per_lb
            _csq = cost_per_sqft
        wh_cost += warehousing_cost(demand, _sq, _csq, fixed_cost)

    total_demand = float(df['DemandLbs'].sum())

    # inbound cost
    inbound_cost = 0.0
    if consider_inbound and inbound_pts:
        if rdc_only_coords:
            inbound_cost = _inbound_cost_to_multiple_rdcs(total_demand, inbound_pts, inbound_rate, rdc_only_coords)
        else:
            inbound_cost = _transfer_cost_multi(inbound_pts, centers, demand_list, inbound_rate)

    # transfer cost
    transfer_cost = 0.0
    if rdc_only_idx and transfer_rate > 0:
        transfer_cost = _transfer_cost_multi_rdcs(rdc_only_coords, centers, rdc_only_idx, demand_list, transfer_rate)

    total = outbound_cost + inbound_cost + transfer_cost + wh_cost
    return {
        'total_cost': total,
        'out_cost': outbound_cost,
        'in_cost': inbound_cost,
        'trans_cost': transfer_cost,
        'wh_cost': wh_cost,
        'assigned': assigned,
        'demand_per_wh': demand_list,
        'rdc_only_idx': rdc_only_idx,
    }

# ─────────────────────────────────────────────────────────────
# optimize over k
# ─────────────────────────────────────────────────────────────
def optimize(
    df,
    k_vals,
    rate_out,
    sqft_per_lb,
    cost_per_sqft,
    fixed_cost,
    *,
    consider_inbound=False,
    inbound_rate=0.0,
    inbound_pts=None,
    fixed_centers=None,
    seed=42,
    rdc_list=None,
    transfer_rate=0.0,
    rdc_sqft_per_lb=None,
    rdc_cost_per_sqft=None,
, candidate_centers=None):
    if fixed_centers:
        fixed_centers = np.asarray(fixed_centers, dtype=float).reshape(-1, 2)
    else:
        fixed_centers = np.empty((0, 2))

    if rdc_list is None:
        rdc_list = []

    coords = df[['Longitude', 'Latitude']].values
    weights = df['DemandLbs'].values
    best = None

    for k in k_vals:
        base_centers = fixed_centers.copy()

        # add SDC‑type RDCs (serve customers)
        for rd in rdc_list:
            if rd['is_sdc']:
                base_centers = np.vstack([base_centers, rd['coords']])

        if k < len(base_centers):
            continue

        k_rem = k - len(base_centers)
        if k_rem == 0:
            centers = base_centers
        else:
            km = KMeans(n_clusters=k_rem, n_init='auto', random_state=seed)
            km.fit(coords, sample_weight=weights)
            centers = np.vstack([base_centers, km.cluster_centers_])

        # append pure RDCs (not serving customers)
        rdc_only_idx = []
        rdc_only_coords = []
        for rd in rdc_list:
            if not rd['is_sdc']:
                rdc_only_idx.append(len(centers))
                rdc_only_coords.append(rd['coords'])
                centers = np.vstack([centers, rd['coords']])

        result = evaluate(
            df,
            centers,
            rate_out,
            sqft_per_lb,
            cost_per_sqft,
            fixed_cost,
            consider_inbound=consider_inbound,
            inbound_rate=inbound_rate,
            inbound_pts=inbound_pts,
            rdc_only_idx=rdc_only_idx,
            rdc_only_coords=rdc_only_coords,
            transfer_rate=transfer_rate,
            rdc_sqft_per_lb=rdc_sqft_per_lb,
            rdc_cost_per_sqft=rdc_cost_per_sqft,
        )

        if (best is None) or (result['total_cost'] < best['total_cost']):
            best = result
            best['k'] = k
            best['centers'] = centers

    return best
