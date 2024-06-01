import numpy as np
import rioxarray as rio
import xarray as xr

from .core import to_crs


def write_crs(ds, x, y, crs):
    """
    Writes the coordinate reference system (CRS) information to a dataset.

    Args:
        ds (xr.Dataset): The dataset to which the CRS information will be added.
        x (str): Name of the x dimension.
        y (str): Name of the y dimension.
        crs (Union[str, int, pyproj.CRS]): The projection to use.

    Returns:
        xr.Dataset: The updated dataset with the CRS information.
    """
    if crs is not None:
        crs = to_crs(crs)
        ds.rio.set_spatial_dims(x_dim=x, y_dim=y, inplace=True)
        ds.rio.write_grid_mapping(inplace=True)
        ds.rio.write_crs(crs, inplace=True)
        ds.rio.write_coordinate_system(inplace=True)
        for k in ds:
            ds[k].encoding.update({'coordinates': f'{y} {x}'})
        # for compatibility with EPSG 27572 and Panoply:
        if crs.to_epsg() == 27572:
            ds["spatial_ref"].attrs["latitude_of_projection_origin"] = 46.8/0.99987742
            ds["spatial_ref"].attrs["longitude_of_central_meridian"] = 2.33722917
    return ds


def write_least_significant_digit(ds, least_significant_digit):
    """
    Sets the number of significant digits to retain when encoding the dataset's data.

    Args:
        ds (xr.Dataset): The dataset to be updated.
        least_significant_digit (Union[int, dict, str]):
            - int: The number of significant digits to retain for all variables.
            - dict: A dictionary mapping the number of significant digits to retain
                     for each variable in the dataset.
            - "auto": Automatically calculates the number of significant digits from the data.

    Returns:
        xr.Dataset: The updated dataset with significant digits information.
    """
    if isinstance(least_significant_digit, int):
        for k in ds:
            ds[k].encoding.update({"least_significant_digit": least_significant_digit})

    if isinstance(least_significant_digit, dict):
        for k in ds:
            if k in least_significant_digit:
                ds[k].encoding.update({"least_significant_digit": least_significant_digit[k]})

    if least_significant_digit == "auto":
        for k in ds:
            m = abs(ds[k].where(ds[k] != 0)).min()
            if m > 0:
                M = int(max(np.ceil(-np.log10(m) + 1), 0))
                ds[k].encoding.update({"least_significant_digit": M})
    return ds


def encode_dataset(ds: xr.Dataset, time="time", y="y", x="x", crs=None, least_significant_digit=None, complevel=6, chunksizes=None,
                   to_float32=True, reset_encoding=True) -> xr.Dataset:
    """
    Encodes the dataset's data with the specified parameters.

    Args:
        ds (xr.Dataset): The dataset to be encoded.
        time (str, optional): Name of the time dimension. Default: "time".
        y (str, optional): Name of the y dimension. Default: "y".
        x (str, optional): Name of the x dimension. Default: "x".
        crs (Union[str, int, pyproj.CRS, None], optional): The projection to use. Default: None.
        least_significant_digit (Union[int, dict, str, None], optional):
            Number of significant digits to retain when encoding the data.
            Default: None.
        complevel (int, optional): The compression level to use. Default: 6.
        chunksizes (Tuple[int, ...], optional): Chunk sizes for parallel processing. Default: None.
        to_float32 (bool, optional): If True, converts floating point variables to float32. Default: True.
        reset_encoding (bool, optional): If True, resets existing encodings. Default: True.

    Returns:
        xr.Dataset: The dataset encoded with the specified parameters.

    Note:
        - This function updates the encoding of the Dataset variables using zlib compression and the compression level specified by "complevel". If "reset_encoding" is True (default), the encoding for each variable is completely reset; otherwise, the encoding is updated without resetting other parameters.
        - The function also updates the encoding of the time coordinate if it exists, specifying the time unit as "hours since 1970-01-01 00:00:00".
        - If "chunksizes" is specified as a tuple, the chunk sizes are updated for each variable in the Dataset.
        - The auxiliary functions write_crs and write_least_significant_digit are called to write the CRS information and the least significant digits to the Dataset.

    Example:
        .. code-block:: python

            ds = xr.Dataset(...)
            encode_dataset(ds)
            ds

        .. code-block:: python

            ds_with_time = xr.Dataset(...)
            ds_encoded = encode_dataset(ds_with_time, time='time', chunksizes=(100, 100))

        .. code-block:: python

            ds_with_crs_and_ls_digit = xr.Dataset(...)
            ds_encoded = encode_dataset(ds_with_crs_and_ls_digit, crs="EPSG:4326", least_significant_digit=2)
    """

    encoding = {"zlib": True, "complevel": complevel}

    for k in ds.variables:
        if ds[k].dtype.kind in 'bfi':
            if reset_encoding:
                ds[k].encoding = encoding
            else:
                ds[k].encoding.update(encoding)
            if ds[k].dtype.kind == 'f' and to_float32:
                encoding = ds[k].encoding
                ds[k] = ds[k].astype(np.float32, keep_attrs=True)
                ds[k].encoding = encoding
        elif reset_encoding:
            ds[k].encoding = {}

    if time in ds:
        ds[time].encoding.update({"units": "hours since 1970-01-01 00:00:00"})

    if isinstance(chunksizes, tuple):
        for k in ds:
            ds[k].encoding.update({'chunksizes': chunksizes})

    write_crs(ds=ds, x=x, y=y, crs=crs)
    write_least_significant_digit(ds=ds, least_significant_digit=least_significant_digit)
    return ds
