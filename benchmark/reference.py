"""Scalar angular Fraunhofer field and tensor-product pixel integration."""
import numpy as np


def field(points_m, areas_m2, opd_m, wavelength_m, theta_x_rad, theta_y_rad,
          amplitude=1., check=lambda: None):
    """Integrate A exp(+2pi i OPD/lambda) with the negative Fourier kernel.

    Output axes are [theta_y, theta_x]; this unnormalized field has units m^2.
    """
    if (not np.isscalar(wavelength_m) or any(np.iscomplexobj(a) for a in
            (points_m, areas_m2, opd_m, wavelength_m, theta_x_rad, theta_y_rad))):
        raise ValueError("coordinates, areas, OPD, wavelength and angles must be real")
    points = np.asarray(points_m, dtype=float)
    areas = np.asarray(areas_m2, dtype=float)
    opd = np.asarray(opd_m, dtype=float)
    tx, ty = np.asarray(theta_x_rad, dtype=float), np.asarray(theta_y_rad, dtype=float)
    amp = np.asarray(amplitude, dtype=complex)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) == 0
            or areas.shape != (len(points),) or opd.shape != areas.shape
            or amp.shape not in ((), areas.shape)
            or tx.ndim != 1 or ty.ndim != 1 or not len(tx) or not len(ty)):
        raise ValueError("incompatible pupil, OPD, amplitude or angular arrays")
    if (not all(np.isfinite(a).all() for a in (points, areas, opd, amp, tx, ty))
            or not np.isfinite(wavelength_m) or wavelength_m <= 0
            or np.any(areas < 0) or areas.sum() <= 0):
        raise ValueError("finite inputs, positive wavelength and nonnegative areas required")
    result = np.zeros((len(ty), len(tx)), dtype=complex)
    strength = areas*amp*np.exp(2j*np.pi*opd/wavelength_m)
    # ponytail: blocked direct quadrature suffices for small reference images; no NUFFT.
    for start in range(0, len(points), 4096):
        p = points[start:start+4096]
        ex = np.exp(-2j*np.pi*np.outer(tx, p[:, 0])/wavelength_m)
        ey = np.exp(-2j*np.pi*np.outer(ty, p[:, 1])/wavelength_m)
        result += (ey*strength[start:start+4096]) @ ex.T
        check()
    return result


def pixel_integral(density, order, weights):
    """Integrate [row, y node, column, x node] angular density; no ROI renormalization."""
    if np.iscomplexobj(density) or np.iscomplexobj(weights):
        raise ValueError("intensity density and pixel weights must be real")
    density, weights = np.asarray(density, dtype=float), np.asarray(weights, dtype=float)
    if (not isinstance(order, (int, np.integer)) or order <= 0
            or density.ndim != 2 or not density.size or weights.shape != (order,)
            or any(n % order for n in density.shape)):
        raise ValueError("pixel density shape must match quadrature order and weights")
    if (not np.isfinite(density).all() or not np.isfinite(weights).all()
            or np.any(density < 0) or np.any(weights <= 0)):
        raise ValueError("finite nonnegative density and positive weights required")
    ny, nx = (n//order for n in density.shape)
    return np.einsum("a,b,iajb->ij", weights, weights, density.reshape(ny, order, nx, order))
