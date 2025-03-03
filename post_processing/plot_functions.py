import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import matplotlib.colors as mcolors

def plot_2d(plotquantity, x_data, y_data, **kwargs):
    """
    This function plots 2D data using matplotlib.

    Parameters:
    plotquantity (array-like): The quantity to be plotted.
    x_data (array-like): The x-coordinates of the data points.
    y_data (array-like): The y-coordinates of the data points.
    xy_center (list, optional): The center of the x and y axes. Default is [0, 0].
    xy_plot_range (list, optional): The range of the x and y axes. Default is [10, 10].
    ylim (list, optional): The limits for the colorbar. Default is None.
    func_str_title (str, optional): The title of the plot. Default is ''.
    func_str_label (str, optional): The label for the colorbar. Default is ''.
    CoordSystem (str, optional): The coordinate system used in the plot. Default is ''.
    plot_gpts (bool, optional): Whether to overplot the original points as white dots. Default is False.
    dpi (int, optional): The resolution of the plot in dots per inch. Default is 120.
    label_fontsize (int, optional): The font size for the labels. Default is 16.
    title_fontsize (int, optional): The font size for the title. Default is 16.
    axes_label (str, optional): The label for the axes. Default is 'xy'.
    cmap (str, optional): The colormap to use for the plot. Default is 'viridis'.
    plot_contours (bool, optional): Whether to plot contour lines. Default is False.
    num_contours (int, optional): The number of contour levels. Default is 30.
    log_scale (bool, optional): Whether to use a logarithmic color scale. Default is False.
    save_path (str, optional): Path to save the plot. Default is None.
    """

    # Raise an error if plotquantity, x_data, or y_data is not provided
    if plotquantity is None or x_data is None or y_data is None:
        raise ValueError("plotquantity, x_data, and y_data must be provided")

    # Set default values for parameters if not provided
    xy_center = kwargs.get("xy_center", [0, 0])
    xy_plot_range = kwargs.get("xy_plot_range", [10, 10])
    ylim = kwargs.get("ylim", None)
    func_str_title = kwargs.get("func_str_title", "")
    func_str_label = kwargs.get("func_str_label", "")
    plot_gpts = kwargs.get("plot_gpts", False)
    dpi = kwargs.get("dpi", 120)
    label_fontsize = kwargs.get("label_fontsize", 16)
    title_fontsize = kwargs.get("title_fontsize", 14)
    axes_label = kwargs.get("axes_label", "xy")
    cmap = kwargs.get("cmap", "viridis")
    plot_contours = kwargs.get("plot_contours", False)
    num_contours = kwargs.get("num_contours", 30)
    log_scale = kwargs.get("log_scale", False)
    save_path = kwargs.get("save_path", None)

    # Set bounds for the x and y axes
    x0, y0 = xy_center
    dx, dy = xy_plot_range

    x_min, x_max = x0 - dx, x0 + dx
    y_min, y_max = y0 - dy, y0 + dy
    pl_xmin, pl_xmax = x_min, x_max
    pl_ymin, pl_ymax = y_max, y_min

    # Compute plot data
    grid_x, grid_y = np.mgrid[pl_xmin:pl_xmax:600j, pl_ymin:pl_ymax:600j]
    points = np.column_stack((x_data, y_data))
    grid = griddata(points, plotquantity, (grid_x, grid_y), method="cubic")

    # Plot the data
    fig, ax = plt.subplots(dpi=dpi)

    if ylim is not None:
        if log_scale:
            norm = mcolors.LogNorm(vmin=ylim[0], vmax=ylim[1])
            im = ax.imshow(
                grid.T,
                extent=(pl_xmin, pl_xmax, pl_ymax, pl_ymin),
                norm=norm,
                cmap=cmap
            )
        else:
            im = ax.imshow(
                grid.T,
                extent=(pl_xmin, pl_xmax, pl_ymax, pl_ymin),
                vmin=ylim[0],
                vmax=ylim[1],
                cmap=cmap
            )
    else:
        if log_scale:
            norm = mcolors.LogNorm()
            im = ax.imshow(grid.T, extent=(pl_xmin, pl_xmax, pl_ymax, pl_ymin), norm=norm, cmap=cmap)
        else:
            im = ax.imshow(grid.T, extent=(pl_xmin, pl_xmax, pl_ymax, pl_ymin), cmap=cmap)

    # Set aspect ratio for both the plot and color bar
    aspect_ratio = np.abs((y_min - y_max) / (x_max - x_min))
    cbar = fig.colorbar(im, shrink=aspect_ratio)

    # Set label for colorbar
    if func_str_label != "":
        cbar.set_label(func_str_label, fontsize=label_fontsize)

    # Set labels and title
    if axes_label == "xy":
        ax.set(xlabel=r"$x$", ylabel=r"$y$")
    elif axes_label == "yz":
        ax.set(xlabel=r"$y$", ylabel=r"$z$")
    else:
        raise ValueError(
            f"axes_label = '{axes_label}' not supported. Available options are: 'xy' and 'yz'"
        )

    ax.set(title=func_str_title)
    ax.title.set_size(title_fontsize)

    ax.xaxis.label.set_size(label_fontsize)
    ax.yaxis.label.set_size(label_fontsize)

    if plot_gpts:
        # Filter points inside the chosen bounds
        mask = (x_data > x_min) & (x_data < x_max) & (y_data > y_min) & (y_data < y_max)
        x_filt = x_data[mask]
        y_filt = y_data[mask]

        # Overplot the original points as white dots
        ax.scatter(x_filt, y_filt, color="white", s=0.01)
        ax.scatter(x_filt, y_filt, marker="x", color="white", s=0.1)

    # Generate contour lines
    if plot_contours:
        CS = ax.contour(grid_x, grid_y, grid, levels=num_contours, colors="white", linewidths=0.5, linestyles="solid")
        # ax.clabel(CS, inline=True, fontsize=10)

    # Save the plot if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()

def interpolate_along_axis(data_2d_path, xmin=-10, xmax=10, dx=0.01, which_quantity="Phi", which_axis="z", interp_method="cubic"):
    """
    Interpolates the specified quantity along a given axis, with support for computing np.sqrt(X).

    :param data_2d_path: Path to the file containing the data (columns: yAxis, zAxis, Phi, Pi, X, T, residual).
    :param xmin: Minimum value of the horizontal axis for interpolation.
    :param xmax: Maximum value of the horizontal axis for interpolation.
    :param dx: Spacing between grid points for interpolation.
    :param which_quantity: Quantity to interpolate ("Phi", "Pi", "X", "T", "residual", or "sqrt(X)").
    :param which_axis: Axis along which to interpolate ("y" or "z").
    :param interp_method: Interpolation method to use ("linear", "cubic", or "nearest").
    :return: Tuple of the selected axis (grid points) and interpolated values.
    :raises ValueError: If the selected axis, quantity, or interpolation method is not recognized.
    """
    # Load data from file
    try:
        data_2d = np.loadtxt(data_2d_path)
    except Exception as e:
        raise ValueError(f"Failed to load data from {data_2d_path}: {e}")

    # Extract columns from data_2d
    yAxis, zAxis, Phi, Pi, X, T, residual = data_2d.T

    # Choose quantity to interpolate
    if which_quantity == "Phi":
        plotquantity = Phi
    elif which_quantity == "Pi":
        plotquantity = Pi
    elif which_quantity == "X":
        plotquantity = X
    elif which_quantity == "T":
        plotquantity = T
    elif which_quantity == "residual":
        plotquantity = residual
    elif which_quantity == "sqrt(X)":
        plotquantity = np.sqrt(X)
    else:
        raise ValueError(f"Quantity {which_quantity} not recognized.")

    # Combine yAxis and zAxis into a 2D array
    points = np.array([yAxis, zAxis]).T

    # Compute number of points and define the interpolation grid
    num_pts = int((xmax - xmin) / dx) + 1
    grid_points = np.linspace(xmin, xmax, num=num_pts)
    grid_other = np.zeros_like(grid_points)

    if which_axis == "y":
        xi = np.array([grid_points, grid_other]).T  # Interpolate along y-axis (z=0)
    elif which_axis == "z":
        xi = np.array([grid_other, grid_points]).T  # Interpolate along z-axis (y=0)
    else:
        raise ValueError(f"Axis {which_axis} not recognized for interpolation.")

    # Validate interpolation method
    valid_methods = ["linear", "cubic", "nearest"]
    if interp_method not in valid_methods:
        raise ValueError(f"Interpolation method {interp_method} not recognized. Choose from {valid_methods}.")

    # Perform interpolation
    interpolated_values = griddata(points, plotquantity, xi, method=interp_method, rescale=True)

    return grid_points, interpolated_values
