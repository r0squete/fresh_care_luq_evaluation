"""
Utility functions for LUQ calculation.
arosquete - 2026/02/13
"""

# Imports
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numba import njit, prange

from config import EPOCH, LAT_MAX, R_EARTH


# Functions
def to_epoch_seconds_datetime64(t64, epoch=EPOCH):
    """
    Convert datetime64 to seconds since epoch.

    Inputs
    ------
    t64   : numpy.datetime64 array
    epoch : numpy.datetime64, reference epoch

    Returns
    -------
    numpy.int64 array
    """
    t64s = t64.astype("datetime64[s]")
    return (t64s - epoch).astype("timedelta64[s]").astype(np.int64)


def load_dataset_files(
    dataset_path, pattern="*.nc", eval_start=None, eval_end=None, tau_max_sec=None
):
    """
    Load and concatenate NetCDF files for a given date range.
    Filenames must contain YYYYMMDD. Loads eval window + tau_max buffer.

    Inputs
    ------
    dataset_path : str or Path
    pattern      : str, glob pattern (default "*.nc")
    eval_start   : str "YYYY-MM-DD", start of evaluation window (required)
    eval_end     : str "YYYY-MM-DD", end of evaluation window (required)
    tau_max_sec  : int, max integration time in seconds — extends end buffer

    Returns
    -------
    xarray.Dataset
    """
    files = sorted(glob.glob(str(Path(dataset_path) / pattern)))
    if not files:
        raise FileNotFoundError(f"No files found in {dataset_path}")

    start_dt = pd.to_datetime(eval_start, utc=True)
    end_dt = pd.to_datetime(eval_end, utc=True)

    # +1 day buffer: _find_index_uniform needs i < n-1, the field must have
    # one timestep beyond t0+τ for the final RK4 k4 evaluation.
    tau_days_buf = int(np.ceil(tau_max_sec / (24 * 3600))) if tau_max_sec else 0
    end_dt_buffered = end_dt + pd.Timedelta(days=tau_days_buf + 1)

    filtered_files = []
    for file in files:
        filename = Path(file).name
        match = re.search(r"(\d{4})(\d{2})(\d{2})", filename)
        if match:
            year, month, day = match.groups()
            file_date = pd.to_datetime(f"{year}-{month}-{day}", utc=True)
            if file_date < start_dt:
                continue
            if file_date > end_dt_buffered:
                continue
            filtered_files.append(file)

    if not filtered_files:
        raise FileNotFoundError("No files found in date range")

    print(f"  Loading {len(filtered_files)} of {len(files)} files")
    return xr.open_mfdataset(filtered_files, combine="by_coords")


def load_drifter_file(drifter_file):
    """
    Load a drifter CSV file (NOAA GDP format, 6-header-line).
    Columns: time, latitude, longitude, ve, vn, err_ve, err_vn,
             sst, err_sst, flg_sst, speed, direction.

    Inputs
    ------
    drifter_file : str or Path

    Returns
    -------
    pandas.DataFrame
    """
    df = pd.read_csv(
        drifter_file,
        skiprows=6,
        sep=",",
        names=[
            "time",
            "latitude",
            "longitude",
            "ve",
            "vn",
            "err_ve",
            "err_vn",
            "sst",
            "err_sst",
            "flg_sst",
            "speed",
            "direction",
        ],
        header=0,
    )
    return df


def prep_dataset(
    ds,
    u_name="ug",
    v_name="vg",
    time_name="time",
    lat_name="lat",
    lon_name="lon",
    keep_uv=False,
):
    """
    Prepare a currents dataset for advection: convert to epoch seconds,
    radians, and angular velocities (rad/s). Handles (time,lon,lat) →
    (time,lat,lon) transposition and [0,360] → [-180,180] normalization.

    Inputs
    ------
    ds        : xarray.Dataset
    u_name    : str, zonal velocity variable name
    v_name    : str, meridional velocity variable name
    time_name : str
    lat_name  : str
    lon_name  : str
    keep_uv   : bool, if True also stores raw u, v (m/s) under keys "u" and "v"

    Returns
    -------
    dict with keys: time_sec, lat_rad, lon_rad, lon_dot, lat_dot
                    and optionally u, v (only if keep_uv=True)
    """
    # coords
    time = ds[time_name].values
    lat = ds[lat_name].values
    lon = ds[lon_name].values

    # absolute seconds since epoch (int64)
    time_sec = to_epoch_seconds_datetime64(time)

    lat_rad = np.deg2rad(lat).astype(np.float32).squeeze()
    lon_rad = np.deg2rad(lon).astype(np.float32).squeeze()

    u = ds[u_name].values.astype(np.float32, copy=False)
    v = ds[v_name].values.astype(np.float32, copy=False)

    # Check if dimensions need transposing before longitude normalization
    if u.shape[1] == len(lon) and u.shape[2] == len(lat):
        print("  INFO: Transposing from (time,lon,lat) to (time,lat,lon)")
        u = u.transpose(0, 2, 1)
        v = v.transpose(0, 2, 1)

    # Normalize longitudes to [-pi, pi] in case dataset uses [0, 2pi] convention
    if lon_rad.max() > np.pi:
        lon_rad = (lon_rad + np.pi).astype(np.float32) % (2 * np.pi) - np.pi
        # After wrapping, sort to ensure monotonically increasing order for interpolation
        sort_idx = np.argsort(lon_rad)
        lon_rad = lon_rad[sort_idx]
        u = u[:, :, sort_idx]
        v = v[:, :, sort_idx]
        print("  INFO: Longitudes converted from [0,360] to [-180,180] and sorted")

    coslat = np.cos(lat_rad)
    coslat = np.maximum(coslat, np.float32(1e-6))

    out = {
        "time_sec": time_sec,
        "lat_rad": lat_rad,
        "lon_rad": lon_rad,
        "lon_dot": (u / (R_EARTH * coslat[None, :, None])).astype(np.float32),  # rad/s
        "lat_dot": (v / R_EARTH).astype(np.float32),  # rad/s
    }
    if keep_uv:
        out["u"] = u
        out["v"] = v
    return out


def prep_drifter(df, time_col="time", lon_col="longitude", lat_col="latitude"):
    """
    Prepare drifter data: convert time to epoch seconds and lon/lat to radians.

    Inputs
    ------
    df        : pandas.DataFrame
    time_col  : str, time column name
    lon_col   : str, longitude column name
    lat_col   : str, latitude column name

    Returns
    -------
    dict with keys: time_sec, lon_rad, lat_rad
    """
    t_pd = pd.to_datetime(df[time_col], utc=True, errors="raise")
    t64 = t_pd.dt.tz_convert(None).to_numpy(dtype="datetime64[s]")

    time_sec = to_epoch_seconds_datetime64(t64)

    lon = df[lon_col].to_numpy(dtype=np.float32)
    lat = df[lat_col].to_numpy(dtype=np.float32)

    lon_rad = np.deg2rad(lon, dtype=np.float32)
    lat_rad = np.deg2rad(lat, dtype=np.float32)

    return {"time_sec": time_sec, "lon_rad": lon_rad, "lat_rad": lat_rad}


def filter_by_temporal_range(
    drifter_data,
    eval_start,
    eval_end,
    tau_max_sec,
    field_start_sec=None,
    field_end_sec=None,
):
    """
    Filter drifter data to the intersection of evaluation and field time ranges.
    eval_start, eval_end and tau_max_sec are required.

    Inputs
    ------
    drifter_data   : dict with keys time_sec, lon_rad, lat_rad
    eval_start     : str "YYYY-MM-DD", evaluation window start (required)
    eval_end       : str "YYYY-MM-DD", evaluation window end (required)
    tau_max_sec    : int, integration window length in seconds (required)
    field_start_sec: int64, field start in epoch seconds (optional)
    field_end_sec  : int64, field end in epoch seconds (optional)

    Returns
    -------
    dict with filtered keys or None if no usable data
    """
    time_sec = drifter_data["time_sec"]

    # Eval window boundaries
    start_dt = pd.to_datetime(eval_start, utc=True)
    eval_start_sec = to_epoch_seconds_datetime64(
        np.array([start_dt.tz_convert(None)], dtype="datetime64[s]")
    )[0]

    end_dt = pd.to_datetime(eval_end, utc=True)
    eval_end_sec = to_epoch_seconds_datetime64(
        np.array([end_dt.tz_convert(None)], dtype="datetime64[s]")
    )[0]

    # Drifter buffer: t0 can reach eval_end and drifter(t0+tau) exists.
    # +86400s (1 day) matches the extra day loaded in load_dataset_files for RK4 k4.
    eval_end_sec = eval_end_sec + int(tau_max_sec) + 86400

    # Most restrictive range: intersection with field coverage.
    start_sec = (
        max(eval_start_sec, field_start_sec)
        if field_start_sec is not None
        else eval_start_sec
    )
    if field_end_sec is not None:
        end_sec = min(eval_end_sec, field_end_sec + int(tau_max_sec) + 86400)
    else:
        end_sec = eval_end_sec

    # Field must extend at least tau_max beyond the usable start
    if field_end_sec is not None:
        if start_sec + tau_max_sec > field_end_sec:
            return None

    mask = (time_sec >= start_sec) & (time_sec <= end_sec)
    if not mask.any():
        return None

    return {
        "time_sec": time_sec[mask],
        "lon_rad": drifter_data["lon_rad"][mask],
        "lat_rad": drifter_data["lat_rad"][mask],
    }


def spherical_distance_km_acos(lon1, lat1, lon2, lat2):
    """
    Compute great-circle distance using spherical law of cosines.

    Inputs
    ------
    lon1, lat1 : float or array, radians
    lon2, lat2 : float, radians

    Returns
    -------
    float or array, distance in km
    """
    # cos(angle) = sinφ1 sinφ2 + cosφ1 cosφ2 cos(Δλ)
    cosang = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(
        lon1 - lon2
    )

    # numerical safety
    cosang = np.clip(cosang, -1.0, 1.0)

    return R_EARTH / 1000 * np.arccos(cosang)


def L_characteristic_km(lon_rad, lat_rad, i0=0, i1=None, dist_fn=None):
    """
    Compute L = max great-circle distance between any two drifter positions.
    This represents the maximum extent of the region that the drifter has visited.

    Inputs
    ------
    lon_rad, lat_rad : 1D arrays (radians)
    i0, i1           : int, window indices (default: full array)
    dist_fn          : callable, distance function returning km
                       (default: spherical_distance_km_acos)

    Returns
    -------
    float, L in km. Returns 0.0 if fewer than 2 points.
    """
    if i1 is None:
        i1 = len(lon_rad)

    if dist_fn is None:
        dist_fn = spherical_distance_km_acos

    lon = lon_rad[i0:i1]
    lat = lat_rad[i0:i1]
    n = len(lon)

    if n < 2:
        return 0.0

    Lmax = 0.0
    # Calculate all pairwise distances efficiently
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_fn(lon[j], lat[j], lon[i], lat[i])
            if d > Lmax:
                Lmax = float(d)

    return Lmax


def make_neighborhood(lon0_rad, lat0_rad, r_km, nx=9, ny=9):
    """
    Create a local square neighborhood centered at (lon0, lat0) with half-size r_km.

    Inputs
    ------
    lon0_rad, lat0_rad : float, center position (radians)
    r_km               : float, half-size of the neighborhood in km
    nx, ny             : int, number of grid points along each axis

    Returns
    -------
    seed_lon_rad, seed_lat_rad : float32 arrays of shape (nx*ny,)
    """
    r_m = float(r_km) * 1000.0

    # Tangent-plane angular half-extents
    coslat = np.cos(float(lat0_rad))
    coslat = max(coslat, 1e-6)  # avoid division by zero near poles

    dlon = np.float32(r_m / (R_EARTH * coslat))  # radians
    dlat = np.float32(r_m / R_EARTH)  # radians

    # Seed grid: offsets in [-dlon, dlon] x [-dlat, dlat]
    xs = np.linspace(-dlon, dlon, nx, dtype=np.float32)
    ys = np.linspace(-dlat, dlat, ny, dtype=np.float32)

    xg, yg = np.meshgrid(xs, ys, indexing="xy")
    seed_lon = np.float32(lon0_rad) + xg
    seed_lat = np.float32(lat0_rad) + yg

    return seed_lon.ravel(), seed_lat.ravel()


@njit
def _find_index_uniform(x, x0, dx, n):
    """
    Return i such that x is inside [x_i, x_{i+1}] for a uniform grid.

    Inputs
    ------
    x  : float, query value
    x0 : float, grid start
    dx : float, grid spacing
    n  : int, number of grid points

    Returns
    -------
    int, index or -1 if out-of-range
    """
    r = (x - x0) / dx  # continuous index
    i = int(np.floor(r))  # lower bracket index

    # out-of-range => invalid
    if i < 0:
        return -1
    if i >= n - 1:  # need i+1 to exist
        return -1
    return i


@njit(parallel=True)
def interp_trilinear_uv(time_sec, lat, lon, u, v, t, lat_q, lon_q):
    """
    Trilinear interpolation (linear in time, lat, lon).

    Inputs
    ------
    time_sec : 1D array [nt] (seconds since epoch 1950, int64)
    lat, lon : 1D arrays [nlat], [nlon] in radians (uniform spacing)
    u, v     : 3D arrays [nt, nlat, nlon] in rad/s (lon_dot, lat_dot from prep_dataset)
    t        : scalar query time (seconds since epoch 1950, float64)
    lat_q,
    lon_q    : query positions (same shape), in radians

    Returns
    -------
    u_q, v_q : interpolated angular velocities (same shape as lat_q/lon_q), rad/s.
               Out-of-domain points return NaN.
    """
    nt = time_sec.size
    nlat = lat.size
    nlon = lon.size

    # grid start + spacing (assumes uniform grids) - convert to float64 for calculations
    t0 = np.float64(time_sec[0])
    dtg = np.float64(time_sec[1] - time_sec[0])
    lat0 = lat[0]
    dlat = lat[1] - lat[0]
    lon0 = lon[0]
    dlon = lon[1] - lon[0]

    # time bracket
    it = _find_index_uniform(np.float64(t), t0, dtg, nt)
    if it < 0:
        # out of time range -> return all NaNs
        u_out = np.empty_like(lat_q)
        v_out = np.empty_like(lat_q)
        for k in prange(lat_q.size):
            u_out.ravel()[k] = np.nan
            v_out.ravel()[k] = np.nan
        return u_out, v_out

    # time weight in [0,1] - use float64 for precision
    wt = (np.float64(t) - (t0 + np.float64(it) * dtg)) / dtg

    # outputs
    u_out = np.empty_like(lat_q)
    v_out = np.empty_like(lat_q)

    # flatten for a single fast loop
    lat_flat = lat_q.ravel()
    lon_flat = lon_q.ravel()
    u_flat = u_out.ravel()
    v_flat = v_out.ravel()

    # main loop (each query point)
    for k in prange(lat_flat.size):
        ph = lat_flat[k]  # query latitude (rad)
        la = lon_flat[k]  # query longitude (rad)

        # spatial brackets
        iy = _find_index_uniform(ph, lat0, dlat, nlat)
        ix = _find_index_uniform(la, lon0, dlon, nlon)

        if iy < 0 or ix < 0:
            # out of spatial range -> NaN
            u_flat[k] = np.nan
            v_flat[k] = np.nan
            continue

        # weights in [0,1]
        wy = (ph - (lat0 + iy * dlat)) / dlat
        wx = (la - (lon0 + ix * dlon)) / dlon

        # 8 corners of the (t,lat,lon) cube
        # time it (t0 slice)
        u000 = u[it, iy, ix]
        u001 = u[it, iy, ix + 1]
        u010 = u[it, iy + 1, ix]
        u011 = u[it, iy + 1, ix + 1]
        # time it+1 (t1 slice)
        u100 = u[it + 1, iy, ix]
        u101 = u[it + 1, iy, ix + 1]
        u110 = u[it + 1, iy + 1, ix]
        u111 = u[it + 1, iy + 1, ix + 1]

        v000 = v[it, iy, ix]
        v001 = v[it, iy, ix + 1]
        v010 = v[it, iy + 1, ix]
        v011 = v[it, iy + 1, ix + 1]

        v100 = v[it + 1, iy, ix]
        v101 = v[it + 1, iy, ix + 1]
        v110 = v[it + 1, iy + 1, ix]
        v111 = v[it + 1, iy + 1, ix + 1]

        # bilinear in (lat,lon) for each time slice
        u0 = (1 - wy) * ((1 - wx) * u000 + wx * u001) + wy * (
            (1 - wx) * u010 + wx * u011
        )
        u1 = (1 - wy) * ((1 - wx) * u100 + wx * u101) + wy * (
            (1 - wx) * u110 + wx * u111
        )

        v0 = (1 - wy) * ((1 - wx) * v000 + wx * v001) + wy * (
            (1 - wx) * v010 + wx * v011
        )
        v1 = (1 - wy) * ((1 - wx) * v100 + wx * v101) + wy * (
            (1 - wx) * v110 + wx * v111
        )

        # linear in time - use float64 weight
        u_flat[k] = (np.float64(1.0) - wt) * u0 + wt * u1
        v_flat[k] = (np.float64(1.0) - wt) * v0 + wt * v1

    return u_out, v_out


@njit
def advect_seeds_rk4(
    time_sec, lat_grid, lon_grid, lon_dot, lat_dot, t0, tau_sec, dt_sec, lon0, lat0
):
    """
    Advect ns seeds from (lon0,lat0) at t0 for tau_sec using fixed-step RK4.

    Inputs
    ------
    time_sec : 1D array [nt] (uniform spacing, seconds since epoch 1950, int64)
    lat_grid, lon_grid : 1D float32 arrays [ny], [nx] in radians (uniform)
    lon_dot, lat_dot : float32 arrays [nt,ny,nx] in rad/s
    t0 : scalar seconds since epoch 1950 (int64 or float64)
    tau_sec : scalar seconds to integrate forward
    dt_sec : scalar time step (e.g. 3600, 7200, 10800)
    lon0, lat0 : float32 arrays [ns] initial seed positions (radians)

    Returns
    -------
    lonT, latT : float32 arrays [ns]
    valid : uint8 array [ns] (1 valid, 0 invalid)
    """
    ns = lon0.size
    lon = lon0.copy()
    lat = lat0.copy()
    valid = np.ones(ns, dtype=np.uint8)

    # reusable temporaries
    lon_a = np.empty(ns, dtype=np.float32)
    lat_a = np.empty(ns, dtype=np.float32)
    lon_b = np.empty(ns, dtype=np.float32)
    lat_b = np.empty(ns, dtype=np.float32)
    lon_c = np.empty(ns, dtype=np.float32)
    lat_c = np.empty(ns, dtype=np.float32)

    nsteps = int(np.ceil(tau_sec / dt_sec))
    t = np.float64(t0)  # convert to float64 for internal calculations

    for step in range(nsteps):
        # shorter last step if needed
        dt = np.float64(dt_sec)
        if step == nsteps - 1:
            rem = np.float64(tau_sec - (nsteps - 1) * dt_sec)
            if rem > 0:
                dt = rem

        half = np.float64(0.5) * dt
        sixth = dt / np.float64(6.0)

        # k1 at (t, lon, lat)
        k1_lon, k1_lat = interp_trilinear_uv(
            time_sec, lat_grid, lon_grid, lon_dot, lat_dot, t, lat, lon
        )

        # A = y + dt/2*k1
        for i in range(ns):
            if valid[i] == 0:
                lon_a[i] = lon[i]
                lat_a[i] = lat[i]
                continue
            if np.isnan(k1_lon[i]) or np.isnan(k1_lat[i]):
                valid[i] = 0
                lon_a[i] = lon[i]
                lat_a[i] = lat[i]
                continue
            la = lat[i] + np.float32(half * k1_lat[i])
            if la > LAT_MAX or la < -LAT_MAX:
                valid[i] = 0
                lon_a[i] = lon[i]
                lat_a[i] = lat[i]
                continue
            lon_a[i] = lon[i] + np.float32(half * k1_lon[i])
            lat_a[i] = la

        # k2 at (t+dt/2, A)
        k2_lon, k2_lat = interp_trilinear_uv(
            time_sec, lat_grid, lon_grid, lon_dot, lat_dot, t + half, lat_a, lon_a
        )

        # B = y + dt/2*k2
        for i in range(ns):
            if valid[i] == 0:
                lon_b[i] = lon[i]
                lat_b[i] = lat[i]
                continue
            if np.isnan(k2_lon[i]) or np.isnan(k2_lat[i]):
                valid[i] = 0
                lon_b[i] = lon[i]
                lat_b[i] = lat[i]
                continue
            lb = lat[i] + np.float32(half * k2_lat[i])
            if lb > LAT_MAX or lb < -LAT_MAX:
                valid[i] = 0
                lon_b[i] = lon[i]
                lat_b[i] = lat[i]
                continue
            lon_b[i] = lon[i] + np.float32(half * k2_lon[i])
            lat_b[i] = lb

        # k3 at (t+dt/2, B)
        k3_lon, k3_lat = interp_trilinear_uv(
            time_sec, lat_grid, lon_grid, lon_dot, lat_dot, t + half, lat_b, lon_b
        )

        # C = y + dt*k3
        for i in range(ns):
            if valid[i] == 0:
                lon_c[i] = lon[i]
                lat_c[i] = lat[i]
                continue
            if np.isnan(k3_lon[i]) or np.isnan(k3_lat[i]):
                valid[i] = 0
                lon_c[i] = lon[i]
                lat_c[i] = lat[i]
                continue
            lc = lat[i] + np.float32(dt * k3_lat[i])
            if lc > LAT_MAX or lc < -LAT_MAX:
                valid[i] = 0
                lon_c[i] = lon[i]
                lat_c[i] = lat[i]
                continue
            lon_c[i] = lon[i] + np.float32(dt * k3_lon[i])
            lat_c[i] = lc

        # k4 at (t+dt, C)
        k4_lon, k4_lat = interp_trilinear_uv(
            time_sec, lat_grid, lon_grid, lon_dot, lat_dot, t + dt, lat_c, lon_c
        )

        # update
        for i in range(ns):
            if valid[i] == 0:
                continue
            if np.isnan(k4_lon[i]) or np.isnan(k4_lat[i]):
                valid[i] = 0
                continue

            dlon = np.float32(
                sixth * (k1_lon[i] + 2.0 * k2_lon[i] + 2.0 * k3_lon[i] + k4_lon[i])
            )
            dlat = np.float32(
                sixth * (k1_lat[i] + 2.0 * k2_lat[i] + 2.0 * k3_lat[i] + k4_lat[i])
            )

            lat_new = lat[i] + dlat
            if lat_new > LAT_MAX or lat_new < -LAT_MAX:
                valid[i] = 0
                continue

            lon[i] = lon[i] + dlon
            lat[i] = lat_new

        t = t + dt

    return lon, lat, valid


def luq_case(
    field,
    t0_sec,
    tau_sec,
    dt_sec,
    lon0_rad,
    lat0_rad,
    lon_target_rad,
    lat_target_rad,
    r_km,
    nx=100,
    ny=100,
    min_valid_frac=0.8,
    return_map=False,
    fill_invalid=np.nan,
):
    """
    Compute LUQ for one case: neighborhood → advect → distance to target → mean.

    Inputs
    ------
    field            : dict, output of prep_dataset
    t0_sec           : int64, seconds since epoch 1950
    tau_sec          : int/float, seconds forward
    dt_sec           : int/float, RK4 time step in seconds (e.g. 3600)
    lon0_rad, lat0_rad : float32, center (drifter at t0), radians
    lon_target_rad, lat_target_rad : float32, target (drifter at t0+tau), radians
    r_km             : float, neighborhood half-size in km
    nx, ny           : int, neighborhood grid size
    min_valid_frac   : float, discard if too many seeds invalid
    return_map       : bool, if True returns full spatial map
    fill_invalid     : value for invalid grid points when return_map=True

    Returns
    -------
    If return_map=False:
        luq_mean_km : float32, NaN if not enough valid
        valid_frac  : float32

    If return_map=True:
        luq_mean_km : float32, NaN if not enough valid
        valid_frac  : float32
        seed_lon_grid : (ny,nx) array in radians
        seed_lat_grid : (ny,nx) array in radians
        LUQ_grid_km   : (ny,nx) array LUQ values in km
        valid_grid    : (ny,nx) bool array
    """
    # 1) neighborhood seeds (float32 arrays)
    seed_lon, seed_lat = make_neighborhood(lon0_rad, lat0_rad, r_km, nx=nx, ny=ny)

    # 2) advect seeds
    lonT, latT, valid = advect_seeds_rk4(
        field["time_sec"],
        field["lat_rad"],
        field["lon_rad"],
        field["lon_dot"],
        field["lat_dot"],
        t0_sec,
        tau_sec,
        dt_sec,
        seed_lon,
        seed_lat,
    )

    valid_mask = valid == 1
    valid_frac = valid_mask.mean().astype(np.float32)

    # 3) distances to target (NaN / fill_invalid for invalid seeds)
    d = np.full(seed_lon.shape, fill_invalid, dtype=np.float32)
    if valid_frac >= min_valid_frac:
        d[valid_mask] = spherical_distance_km_acos(
            lonT[valid_mask], latT[valid_mask], lon_target_rad, lat_target_rad
        ).astype(np.float32)
        # 4) mean over valid seeds
        luq_mean_km = np.nanmean(d[valid_mask]).astype(np.float32)
    else:
        luq_mean_km = np.float32(np.nan)

    # Return based on flag
    if not return_map:
        return luq_mean_km, np.float32(valid_frac)
    else:
        # Reshape to grids for plotting
        seed_lon_grid = seed_lon.reshape(ny, nx)
        seed_lat_grid = seed_lat.reshape(ny, nx)
        LUQ_grid = d.reshape(ny, nx)
        valid_grid = valid_mask.reshape(ny, nx)

        return (
            luq_mean_km,
            np.float32(valid_frac),
            seed_lon_grid,
            seed_lat_grid,
            LUQ_grid,
            valid_grid,
        )


def save_results_csv(results, output_dir, tau_days, r_kms, eval_start=None):
    """
    Write LUQ results to CSV, one file per drifter.
    Layout: all summary rows first, then all timeseries rows below.

    Inputs
    ------
    results    : dict, output of run_evaluation
    output_dir : str or Path
    tau_days   : list of int
    r_kms      : list of int
    eval_start : str "YYYY-MM-DD", used in output filename
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for drifter_idx, drifter_results in results.items():
        if not isinstance(drifter_idx, int):  # Skip metadata entries
            continue

        filename = drifter_results["filename"]

        # Get date string for output filename
        date_str = eval_start if eval_start is not None else "xxx"

        all_rows = []
        timeseries_rows = []
        for dataset_name in drifter_results.keys():
            if dataset_name == "filename":
                continue
            for r_km in r_kms:
                for tau_day in tau_days:
                    tau_sec = np.int64(tau_day * 24 * 3600)
                    if r_km not in drifter_results[dataset_name]:
                        continue
                    if tau_sec not in drifter_results[dataset_name][r_km]:
                        continue
                    data = drifter_results[dataset_name][r_km][tau_sec]
                    L_char = data["L_characteristic_km"]
                    # Summary row
                    all_rows.append(
                        {
                            "data_type": "summary",
                            "drifter_id": drifter_idx,
                            "drifter_filename": filename,
                            "dataset": dataset_name,
                            "r_km": r_km,
                            "tau_days": tau_day,
                            "t0_datetime": "",
                            "luq_km": data["luq_mean_km"],
                            "luq_normalized": data["luq_normalized"],
                            "valid_frac": data["valid_frac"],
                            "n_cases": data["n_cases"],
                            "L_characteristic_km": L_char,
                        }
                    )
                    # Collect timeseries rows separately
                    for calc in data.get("individual_calculations", []):
                        timeseries_rows.append(
                            {
                                "data_type": "timeseries",
                                "drifter_id": drifter_idx,
                                "drifter_filename": filename,
                                "dataset": dataset_name,
                                "r_km": r_km,
                                "tau_days": tau_day,
                                "t0_datetime": calc["t0_datetime"].strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                                "lon0_deg": calc["lon0_deg"],
                                "lat0_deg": calc["lat0_deg"],
                                "luq_km": calc["luq_km"],
                                "luq_normalized": calc["luq_normalized"],
                                "valid_frac": calc["valid_frac"],
                                "n_cases": "",
                                "L_characteristic_km": L_char,
                            }
                        )
        # Summary rows first, then all timeseries rows below
        all_rows.extend(timeseries_rows)

        if all_rows:
            df = pd.DataFrame(all_rows)
            output_file = (
                output_path
                / f"luq_results_drifter_{filename.replace('.csv', '')}_{date_str}.csv"
            )
            df.to_csv(
                output_file,
                index=False,
                float_format="%.6f",
                sep=";",
                encoding="utf-8-sig",
            )
            n_summary = len(all_rows) - len(timeseries_rows)
            print(
                f"  ✓ {output_file.name} ({n_summary} summary + {len(timeseries_rows)} timeseries records)"
            )


def compute_M(field, t0_sec, tau_sec, dt_sec, lon_grid_rad, lat_grid_rad):
    """
    Compute Lagrangian descriptor M on a grid of initial conditions.
    Forward integration only, from t0_sec over tau_sec.
    Step metric: spherical arc-length sqrt(dlat² + (cos(lat_mid)·dlon)²).

    Inputs
    ------
    field        : dict, output of prep_dataset
    t0_sec       : int64, integration start in epoch seconds
    tau_sec      : int/float, integration window in seconds
    dt_sec       : int/float, time step in seconds
    lon_grid_rad : 1D float32 array, longitude grid in radians
    lat_grid_rad : 1D float32 array, latitude grid in radians

    Returns
    -------
    M : float64 array (nlat, nlon)
    """
    nsteps = int(np.ceil(tau_sec / dt_sec))
    nlat = len(lat_grid_rad)
    nlon = len(lon_grid_rad)

    # Initial condition meshgrid: lat[j, i], lon[j, i]
    lon_mesh, lat_mesh = np.meshgrid(lon_grid_rad, lat_grid_rad)  # (nlat, nlon)
    lon = lon_mesh.astype(np.float32)
    lat = lat_mesh.astype(np.float32)

    M_acc = np.zeros((nlat, nlon), dtype=np.float64)
    t = np.float64(t0_sec)

    for step in range(nsteps):
        dt = np.float64(dt_sec)
        if step == nsteps - 1:
            rem = np.float64(tau_sec - (nsteps - 1) * dt_sec)
            if rem > 0:
                dt = rem

        # RK4 stages using interp_trilinear_uv (lon_dot, lat_dot in rad/s).
        # NaN (land/out-of-domain) → 0: particle stays put that step.
        k1_lon, k1_lat = interp_trilinear_uv(
            field["time_sec"],
            field["lat_rad"],
            field["lon_rad"],
            field["lon_dot"],
            field["lat_dot"],
            t,
            lat,
            lon,
        )
        k1_lon = np.nan_to_num(k1_lon)
        k1_lat = np.nan_to_num(k1_lat)

        k2_lon, k2_lat = interp_trilinear_uv(
            field["time_sec"],
            field["lat_rad"],
            field["lon_rad"],
            field["lon_dot"],
            field["lat_dot"],
            t + 0.5 * dt,
            lat + np.float32(0.5 * dt) * k1_lat,
            lon + np.float32(0.5 * dt) * k1_lon,
        )
        k2_lon = np.nan_to_num(k2_lon)
        k2_lat = np.nan_to_num(k2_lat)

        k3_lon, k3_lat = interp_trilinear_uv(
            field["time_sec"],
            field["lat_rad"],
            field["lon_rad"],
            field["lon_dot"],
            field["lat_dot"],
            t + 0.5 * dt,
            lat + np.float32(0.5 * dt) * k2_lat,
            lon + np.float32(0.5 * dt) * k2_lon,
        )
        k3_lon = np.nan_to_num(k3_lon)
        k3_lat = np.nan_to_num(k3_lat)

        k4_lon, k4_lat = interp_trilinear_uv(
            field["time_sec"],
            field["lat_rad"],
            field["lon_rad"],
            field["lon_dot"],
            field["lat_dot"],
            t + dt,
            lat + np.float32(dt) * k3_lat,
            lon + np.float32(dt) * k3_lon,
        )
        k4_lon = np.nan_to_num(k4_lon)
        k4_lat = np.nan_to_num(k4_lat)

        dlon = np.float32(dt / 6.0) * (k1_lon + 2 * k2_lon + 2 * k3_lon + k4_lon)
        dlat = np.float32(dt / 6.0) * (k1_lat + 2 * k2_lat + 2 * k3_lat + k4_lat)

        lon_new = lon + dlon
        lat_new = lat + dlat

        # Spherical arc-length of this step
        lat_mid = 0.5 * (lat + lat_new)
        step_arc = np.sqrt(dlat**2 + (np.cos(lat_mid) * dlon) ** 2)

        # Freeze bad points (out-of-domain or NaN): contribute 0 to M
        bad = ~np.isfinite(step_arc) | ~np.isfinite(lon_new) | ~np.isfinite(lat_new)
        M_acc += np.where(bad, 0.0, step_arc).astype(np.float64)
        lon = np.where(bad, lon, lon_new)
        lat = np.where(bad, lat, lat_new)
        t += dt

    return M_acc
