from typing import Generator

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon


def covering_mesh(gdf: gpd.GeoDataFrame, cell_size: float, return_xy: bool = False, round: int = None, return_indices: bool = False) -> gpd.GeoDataFrame:
    """
    Generates a regular grid covering the bounding box of a GeoDataFrame.

    Parameters:
    - gdf (gpd.GeoDataFrame): Input GeoDataFrame.
    - cell_size (float): Size of grid cells.
    - return_xy (bool, optional): Whether to return x, y coordinates along with the grid. Default is False.
    - round (int, optional): Number of decimal places to round grid coordinates to. Default is None.
    - return_indices (bool, optional): Whether to return grid indices. Default is False.

    Returns:
    - gpd.GeoDataFrame or Tuple[np.ndarray, np.ndarray, gpd.GeoDataFrame]: If `return_xy` is True, returns x, y coordinates and the grid as a GeoDataFrame. Otherwise, returns only the grid.

    """
    xmin, ymin, xmax, ymax = gdf.total_bounds
    x = np.arange(start=xmin, stop=xmax+cell_size, step=cell_size)
    y = np.arange(start=ymin, stop=ymax+cell_size, step=cell_size)
    x = x - x.mean() + (xmin + xmax)/2.
    y = y - y.mean() + (ymin + ymax)/2.
    if isinstance(round, int):
        x, y = np.round(x, round), np.round(y, round)
    xx, yy = np.meshgrid(x, y)
    grid = generate_grid(xx, yy, return_indices=return_indices, crs=gdf.crs)
    if return_xy:
        return x, y, grid
    else:
        return grid


def generate_grid(grid_x: np.ndarray, grid_y: np.ndarray, return_indices: bool = False, crs=None) -> gpd.GeoDataFrame:
    """
    Generates a grid of polygons from arrays of x and y coordinates.

    Parameters:
    - grid_x (np.ndarray): Array of x coordinates.
    - grid_y (np.ndarray): Array of y coordinates.
    - return_indices (bool, optional): Whether to return grid indices. Default is False.
    - crs: Coordinate reference system for the GeoDataFrame. Default is None.

    Returns:
    - gpd.GeoDataFrame: GeoDataFrame containing grid polygons.

    Raises:
    - ValueError: If the shapes of `grid_x` and `grid_y` are not compatible.

    """
    gx, gy = np.array(grid_x), np.array(grid_y)
    if (gx.shape != gy.shape) or (gx.ndim != 2):
        raise ValueError("`grid_x` and `grid_y` must be two 2D arrays of the same shape")

    def _interp_grid(X):
        dX = np.diff(X, axis=1)/2.
        X = np.hstack((X[:, [0]] - dX[:, [0]],
                       X[:, :-1] + dX,
                       X[:, [-1]] + dX[:, [-1]]))
        return X

    ny, nx = gx.shape

    x = _interp_grid(_interp_grid(gx).T).T
    y = _interp_grid(_interp_grid(gy).T).T

    gdf_grid = []

    for i in range(ny):
        for j in range(nx):
            p = Polygon([[x[i, j], y[i, j]], [x[i+1, j], y[i+1, j]], [x[i+1, j+1], y[i+1, j+1]], [x[i, j+1], y[i, j+1]]])
            gdf_grid.append(p)

    gdf_grid = gpd.GeoDataFrame(geometry=gdf_grid)
    if crs:
        gdf_grid = gdf_grid.set_crs(crs)

    if return_indices:
        iy, ix = np.indices(dimensions=(ny, nx))
        gdf_grid = gdf_grid.reset_index()
        gdf_grid['iy'] = iy.ravel()
        gdf_grid['ix'] = ix.ravel()

    return gdf_grid


def generator_geopandas(path: str, batch_size: int = 25_000, max_row: int = None, **kwargs) -> Generator[gpd.GeoDataFrame, None, None]:
    """
    Generates batches of GeoDataFrame from a GeoPackage file.

    Parameters:
    - path (str): Path to the GeoPackage file.
    - batch_size (int): Number of rows to read per batch. Default is 25,000.
    - max_row (int, optional): Maximum number of rows to read. Default is None, meaning all rows will be read.
    - **kwargs: Additional keyword arguments to pass to `geopandas.read_file`.

    Yields:
    - gpd.GeoDataFrame: A batch of GeoDataFrame.

    Raises:
    - ValueError: If batch_size is not a positive integer.

    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    i, j = 0, batch_size
    max_row = np.inf if max_row is None else max_row

    while i < max_row:
        gdf = gpd.read_file(path, rows=slice(i, j), **kwargs)
        yield gdf

        if len(gdf) < batch_size or j >= max_row:
            break

        i += batch_size
        j += batch_size
