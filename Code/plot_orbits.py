import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- Helper Functions ---

def _get_orbit_points(a, e, theta_array=None):
    """
    Generates perifocal coordinates (x_p, y_p, z_p) for an orbit.

    Args:
        a (float): Semi-major axis (km).
        e (float): Eccentricity.
    theta_array (np.array or float, optional): True anomalies (radians).

    Returns:
        tuple: (x_p, y_p, z_p) coordinate arrays.
    """
    if theta_array is None:
        theta_array = np.linspace(0, 2 * np.pi, 400)

    theta_array = np.atleast_1d(theta_array)

    # Polar equation for an ellipse
    r = (a * (1 - e**2)) / (1 + e * np.cos(theta_array))
    
    # Convert to Cartesian coordinates in the orbital (perifocal) plane
    x_p = r * np.cos(theta_array)
    y_p = r * np.sin(theta_array)
    z_p = np.zeros_like(x_p)
    
    return x_p, y_p, z_p

def _get_rotation_vectors(omega, i, raan):
    """
    Calculates the P, Q, W unit vectors (perifocal frame) in ECI coordinates.

    Args:
        omega (float): Argument of perigee (degrees).
        i (float): Inclination (degrees).
        raan (float): Right ascension of the ascending node (degrees).

    Returns:
        tuple: (P, Q, W) unit vectors as 3-element np.arrays.
    """
    # Convert all angles to radians
    omega_rad = np.deg2rad(omega)
    i_rad = np.deg2rad(i)
    raan_rad = np.deg2rad(raan)
    
    # Define sines and cosines
    cos_w = np.cos(omega_rad)
    sin_w = np.sin(omega_rad)
    cos_i = np.cos(i_rad)
    sin_i = np.sin(i_rad)
    cos_R = np.cos(raan_rad)
    sin_R = np.sin(raan_rad)
    
    # P vector (points to perigee)
    P = np.array([
        cos_R * cos_w - sin_R * cos_i * sin_w,
        sin_R * cos_w + cos_R * cos_i * sin_w,
        sin_i * sin_w
    ])
    
    # Q vector (in orbital plane, 90 deg from P)
    Q = np.array([
        -cos_R * sin_w - sin_R * cos_i * cos_w,
        -sin_R * sin_w + cos_R * cos_i * cos_w,
        sin_i * cos_w
    ])
    
    # W vector (normal to orbital plane)
    W = np.array([
        sin_R * sin_i,
        -cos_R * sin_i,
        cos_i
    ])
    
    return P, Q, W

def _transform_to_eci(x_p, y_p, P_vec, Q_vec):
    """
    Transforms perifocal coordinates to ECI coordinates.

    Args:
        x_p (np.array): Perifocal x-coordinates.
        y_p (np.array): Perifocal y-coordinates.
        P_vec (np.array): P unit vector.
        Q_vec (np.array): Q unit vector.

    Returns:
        tuple: (x, y, z) ECI coordinate arrays.
    """
    # P_vec and Q_vec are column vectors, but we need to multiply
    # them by the scalar arrays x_p and y_p.
    # We can do this by (P.reshape(-1, 1) * x_p)
    x = P_vec[0] * x_p + Q_vec[0] * y_p
    y = P_vec[1] * x_p + Q_vec[1] * y_p
    z = P_vec[2] * x_p + Q_vec[2] * y_p
    
    return x, y, z

def _plot_sphere(ax, r, color, alpha=0.8, zorder=1):
    """
    Plots a sphere at the origin.

    Args:
        ax (matplotlib.axes.Axes): The 3D axes object.
        r (float): Radius of the sphere.
        color (str): Color of the sphere.
    """
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, zorder=zorder)

def _plot_plane(ax, r, color, P_vec=None, Q_vec=None):
    """
    Plots a circular plane (Equatorial or Orbital).

    Args:
        ax (matplotlib.axes.Axes): The 3D axes object.
        r (float): Radius of the plane.
        color (str): Color of the plane.
        P_vec (np.array, optional): P vector for orbital plane.
        Q_vec (np.array, optional): Q vector for orbital plane.
    """
    theta = np.linspace(0, 2 * np.pi, 100)
    x_p = r * np.cos(theta)
    y_p = r * np.sin(theta)
    
    if P_vec is not None and Q_vec is not None:
        # Orbital Plane
        x, y, z = _transform_to_eci(x_p, y_p, P_vec, Q_vec)
    else:
        # Equatorial Plane
        x, y, z = x_p, y_p, np.zeros_like(x_p)
        
    verts = [list(zip(x, y, z))]
    poly = Poly3DCollection(verts, alpha=0.2, facecolors=color, zorder=0)
    ax.add_collection3d(poly)
    ax.plot(x, y, z, color=color, alpha=0.4, zorder=0) # Edge

def _plot_arc_3d(ax, center, v1, v2, normal, radius, color, style='-'):
    """
    Plots a 3D arc from vector v1 to v2.

    Args:
        ax (matplotlib.axes.Axes): The 3D axes object.
        center (np.array): Arc center (origin).
        v1 (np.array): Start vector.
        v2 (np.array): End vector.
        normal (np.array): Normal vector defining the arc's plane.
        radius (float): Radius of the arc.
        color (str): Color of the arc.
    """
    # Normalize
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    normal = normal / np.linalg.norm(normal)
    
    # Angle between vectors using atan2 to detect reflex angles
    dot_prod = np.dot(v1, v2)
    cross_prod = np.dot(np.cross(v1, v2), normal)
    angle = np.arctan2(cross_prod, dot_prod)
    if angle < 0:
        angle += 2 * np.pi
    
    # Points for the arc
    t = np.linspace(0, angle, 100)
    
    # Use Rodrigues' rotation formula
    v_cross = np.cross(normal, v1)
    
    arc_points = []
    for ti in t:
        pt = center + radius * (v1 * np.cos(ti) + v_cross * np.sin(ti))
        arc_points.append(pt)
            
    arc_points = np.array(arc_points)
    ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], 
            color=color, linestyle=style, zorder=8)
    
    # Return midpoint for label
    mid_point = arc_points[len(arc_points) // 2]
    return mid_point


def plot_orbit_3d(ax, params, title, is_right_plot=False):
    """
    Generates a complete 3D plot of a satellite orbit.

    Args:
        ax (matplotlib.axes.Axes): The 3D axes object to plot on.
        params (dict): Dictionary of orbital parameters:
                       'a', 'e', 'i', 'raan', 'omega', 'theta_sat'.
        title (str): The title for the subplot.
        is_right_plot (bool): Whether this is the right plot (for label positioning).
    """
    a = params['a']
    e = params['e']
    i = params['i']
    raan = params['raan']
    omega = params['omega']
    theta_sat = params['theta_sat'] # This is 'v' in the image
    
    # --- 1. Setup ---
    ax.set_title(title, fontsize=16, pad=20)
    
    # --- 2. Get Rotation & ECI Vectors ---
    P_vec, Q_vec, W_vec = _get_rotation_vectors(omega, i, raan)
    v_Z = np.array([0, 0, 1])
    v_X = np.array([1, 0, 0])
    
    # --- 3. Plot Earth ---
    earth_r = 6371  # Mean Earth radius in km
    _plot_sphere(ax, earth_r, 'royalblue', alpha=0.1)
    
    # --- 4. Plot ECI Frame ---
    max_dist = a * (1 + e) * 1.5  # Set plot limits
    arrow_len = max_dist * 0.7
    ax.quiver(0, 0, 0, arrow_len, 0, 0, color='k', 
              arrow_length_ratio=0.05, zorder=10)
    ax.quiver(0, 0, 0, 0, arrow_len, 0, color='k', 
              arrow_length_ratio=0.05, zorder=10)
    ax.quiver(0, 0, 0, 0, 0, arrow_len, color='k', 
              arrow_length_ratio=0.05, zorder=10)
    
    ax.text(arrow_len, 0, 0, 'X (Vernal Equinox)', zorder=10)
    ax.text(0, arrow_len, 0, 'Y', zorder=10)
    ax.text(0, 0, arrow_len, 'Z', zorder=10)
    
    # --- 5. Plot Planes ---
    plane_r = a * (1 + e) * 1.2
    # Equatorial Plane
    _plot_plane(ax, plane_r, 'darkcyan')
    ax.text(plane_r, 0, 0, 'Equatorial\nPlane', 
            color='darkcyan', ha='center', va='center')
    # Orbital Plane
    _plot_plane(ax, plane_r, 'gray', P_vec, Q_vec)
    diag_dir = np.array([1.0, 0.0, 1.0])
    diag_dir /= np.linalg.norm(diag_dir)
    diag_on_plane = diag_dir - np.dot(diag_dir, W_vec) * W_vec
    if np.linalg.norm(diag_on_plane) < 1e-6:
        diag_on_plane = P_vec
    diag_on_plane /= np.linalg.norm(diag_on_plane)
    label_pt_orb = plane_r * diag_on_plane
    ax.text(label_pt_orb[0], label_pt_orb[1], label_pt_orb[2], 'Orbital\nPlane', 
            color='gray', ha='center', va='center')
    
    # --- 6. Plot Orbit ---
    x_p, y_p, z_p = _get_orbit_points(a, e)
    x, y, z = _transform_to_eci(x_p, y_p, P_vec, Q_vec)
    ax.plot(x, y, z, color='red', linewidth=2, label='Orbit', zorder=5)
    
    # --- 7. Plot Key Points & Vectors ---
    
    # Perigee (theta = 0)
    x_p_p, y_p_p, _ = _get_orbit_points(a, e, 0)
    v_perigee = np.array(_transform_to_eci(x_p_p, y_p_p, P_vec, Q_vec)).flatten()
    ax.plot([0, v_perigee[0]], [0, v_perigee[1]], [0, v_perigee[2]],
        color='green', linewidth=2, alpha=0.8, zorder=6)
    ax.text(v_perigee[0]*1.1, v_perigee[1]*1.1, v_perigee[2]*1.1, 
            'Perigee', color='green', zorder=10, weight='bold')

    # Apogee (theta = pi)
    x_p_a, y_p_a, _ = _get_orbit_points(a, e, np.pi)
    v_apogee = np.array(_transform_to_eci(x_p_a, y_p_a, P_vec, Q_vec)).flatten()
    ax.text(v_apogee[0]*1.1, v_apogee[1]*1.1, v_apogee[2]*1.1, 
            'Apogee', color='purple', zorder=10, weight='bold')

    # Line of Nodes (connecting Asc/Desc nodes)
    # Vector for Ascending Node
    v_asc_node_unit = np.array([np.cos(np.deg2rad(raan)), 
                                np.sin(np.deg2rad(raan)), 0])
    # Find intersection radius
    r_asc = (a * (1 - e**2)) / (1 + e * np.cos(np.deg2rad(-omega))) # theta = -omega
    v_asc_node_vec = v_asc_node_unit * r_asc
    v_desc_node_vec = -v_asc_node_vec * ( (1 + e * np.cos(np.deg2rad(-omega))) / 
                                          (1 + e * np.cos(np.deg2rad(180 - omega))) )
    
    ax.plot([v_desc_node_vec[0], v_asc_node_vec[0]], 
            [v_desc_node_vec[1], v_asc_node_vec[1]], 
            [v_desc_node_vec[2], v_asc_node_vec[2]], 
            'b--', zorder=4, alpha=0.7)
    ax.text(v_asc_node_vec[0]*1.1, v_asc_node_vec[1]*1.1, 
            v_asc_node_vec[2]*1.1, 'Ascending Node', 
            color='blue', zorder=10, weight='bold')

    # Satellite
    x_p_s, y_p_s, _ = _get_orbit_points(a, e, np.deg2rad(theta_sat))
    v_sat = np.array(_transform_to_eci(x_p_s, y_p_s, P_vec, Q_vec)).flatten()
    ax.plot([0, v_sat[0]], [0, v_sat[1]], [0, v_sat[2]], 'r-', zorder=6, alpha=0.8)
    ax.scatter(v_sat[0], v_sat[1], v_sat[2], color='red', 
               s=100, zorder=7, label='Satellite', edgecolors='k')
    # Adjust label position based on plot side
    if is_right_plot:
        ax.text(v_sat[0]*1.15, v_sat[1]*1.15, v_sat[2]*1.45, 
                'Satellite', color='red', zorder=10, weight='bold')
    else:
        ax.text(v_sat[0]*1.1, v_sat[1]*1.1, v_sat[2]*1.1, 
                'Satellite', color='red', zorder=10, weight='bold')
    sat_radius = np.linalg.norm(v_sat)
    
    # --- 8. Plot Angles ---
    arc_r = max_dist * 0.25
    
    # Plot i (Inclination)
    # The arc starts from a specified point in the equatorial plane and sweeps
    # up to the orbital plane.
    v_start_eq = np.array([7500, -7500, 0])
    v_start_eq = v_start_eq / np.linalg.norm(v_start_eq) # Normalize

    # The end vector is the start vector projected onto the orbital plane.
    # Projection of v_start_eq onto W_vec (normal of orbital plane)
    proj_on_W = np.dot(v_start_eq, W_vec) * W_vec
    # Subtract projection to get vector in the orbital plane
    v_end_orb = v_start_eq - proj_on_W
    v_end_orb = v_end_orb / np.linalg.norm(v_end_orb) # Normalize

    # The normal to the arc's plane is perpendicular to both start and end vectors.
    arc_normal = np.cross(v_start_eq, v_end_orb)
    if np.linalg.norm(arc_normal) < 1e-9: # Handle case where vectors are collinear
        arc_normal = np.cross(v_start_eq, W_vec)

    mid_i = _plot_arc_3d(ax, [0,0,0], v_start_eq, v_end_orb,
                         arc_normal, arc_r, 'orange', style='--')
    ax.text(mid_i[0]*1.2, mid_i[1]*1.2, mid_i[2]*1.2,
            '$i$', color='orange', fontsize=14)

    # Plot $\Omega$ (blue, matching Ascending Node)
    mid_R = _plot_arc_3d(ax, [0,0,0], v_X, v_asc_node_unit, 
                         v_Z, arc_r, 'blue', style='--')
    ax.text(mid_R[0]*1.2, mid_R[1]*1.2, mid_R[2]*1.2, 
            '$Ω$', color='blue', fontsize=14)

    # Plot $\omega$
    v_perigee_unit = v_perigee / np.linalg.norm(v_perigee)
    mid_w = _plot_arc_3d(ax, [0,0,0], v_asc_node_unit, v_perigee_unit, 
                         W_vec, arc_r, 'green', style='--')
    ax.text(mid_w[0]*1.2, mid_w[1]*1.2, mid_w[2]*1.2, 
            '$ω$', color='green', fontsize=14)
    
    # Plot theta (True Anomaly)
    v_sat_unit = v_sat / np.linalg.norm(v_sat)
    theta_radius = sat_radius * 0.7
    mid_theta = _plot_arc_3d(ax, [0,0,0], v_perigee_unit, v_sat_unit, 
                             W_vec, theta_radius, 'red', style='-')
    ax.text(mid_theta[0], mid_theta[1], mid_theta[2], 
            r'$\theta$', color='red', fontsize=14, weight='bold')

    # --- 9. Final Touches ---
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_zlabel('Z (km)', fontsize=12)
    
    # Set limits
    ax.set_xlim([-max_dist, max_dist])
    ax.set_ylim([-max_dist, max_dist])
    ax.set_zlim([-max_dist, max_dist])
    
    # Add parameter box
    # Using 'theta' for true anomaly in the box, as 'v' is on the plot
    param_text = (
        f"Measurements:\n"
        f"$a$ = {a} km\n"
        f"$e$ = {e:.5f}\n"
        f"$i$ = {i:.2f}°\n"
        f"$Ω$ = {raan:.2f}°\n"
        f"$ω$ = {omega:.2f}°\n"
        f"$\\theta$ = {theta_sat:.2f}°"
    )
    ax.text2D(0.02, 0.98, param_text, transform=ax.transAxes, 
              ha='left', va='top', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Set view angle similar to reference
    # Look from a diagonal between the X and Y axes for better orbit visibility
    ax.view_init(elev=25, azim=45)
    ax.dist = 10 # Zoom out a bit

# --- Main Execution ---

if __name__ == "__main__":
    
    # Data from your report
    estimated_params = {
        "a": 6938,          # Section 4.1
        "e": 0.0148,        # Section 5.1
        "i": 52.82,         # Section 5.2
        "raan": 171.93,     # Section 5.3 (from TLE)
        "omega": 95.01,     # Section 5.4 (from TLE)
        "theta_sat": 321.55 # Section 5.5 (Propagated M_e, as proxy for theta)
    }

    # Data from TLE (Table 4)
    tle_params = {
        "a": 6917,
        "e": 0.0001133,
        "i": 53.21,
        "raan": 171.93,
        "omega": 95.01,
        "theta_sat": 265.11 # TLE M_e, as proxy for theta
    }

    # Create the figure and axes
    fig = plt.figure(figsize=(22, 11))
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_orbit_3d(ax1, estimated_params, "Estimated Orbit (Based on Report)", is_right_plot=False)
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_orbit_3d(ax2, tle_params, "TLE Orbit (Ground Truth)", is_right_plot=True)
    
    fig.suptitle('Starlink-3988 Orbital Visualization', fontsize=20, y=0.98)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()