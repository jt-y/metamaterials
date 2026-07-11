import os
import pickle
import glob
import skimage
import skimage.transform
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from os import listdir
from os.path import isfile, join
import shutil, os
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib import patches as ptc
import stmpy

import sys

sys.path.append('../../functions')



def preprocess_chirp(setno, data_dir, frequencies, targetfile, title = 'scppos_'):
    fnames = list(sorted(glob.glob(os.path.join(data_dir, "*.pkl"))))
    
    print('Found %s records' % len(fnames))
    
    data = [[] for _ in frequencies]
    
    for fname in fnames:
        with open(fname, 'rb') as f:
            chirp_data = pickle.load(f)
               
        x, y = [float(coord) for coord in os.path.basename(fname).replace('.pkl', '').replace(title, '').split('_')]
        
        for point in chirp_data:
            freq_index = np.where(frequencies == point[0])[0][0] 
            data[freq_index].append((x, y, point[1], point[2]))
                
    if not data:
        raise RuntimeError('No Data Found')  
        
    for freq in frequencies:
        index = np.where(frequencies == freq)[0][0]       
        np.array(data[index]).dump(targetfile + '_' + str(freq) + '_' + str(setno) + '.pkl')
    
    print('Preprocessing complete!')


def central_cut(real_data, source_pos, edge_relax):
    """
    Get the central hexagonal part of the real space data.
    real_data: [[x_coord, y_coord, amp, phase], ...].
    Also shift the origin to the source position
    """
    
    x = real_data[:, 0] - source_pos[0]
    y = real_data[:, 1] - source_pos[1]
    amp = real_data[:, 2]
    phase = real_data[:, 3]

    x_unit = 10*np.sqrt(3)/4*1e-3
    y_unit = 7.5*1e-3

    # Get the central hexagon
    y_mask = (np.abs(y) <= y_unit*8.2) 
    xy_mask_1 = (np.abs(y+y_unit/x_unit*x) <= y_unit*16+edge_relax)
    xy_mask_2 = (np.abs(y-y_unit/x_unit*x) <= y_unit*16+edge_relax)
    source_mask = (np.sqrt(x**2 + y**2) > edge_relax)  # Avoid the source point
    mask = y_mask & xy_mask_1 & xy_mask_2 & source_mask


    return np.column_stack((x[mask], y[mask], amp[mask], phase[mask]))



def return_sublat_triangle(origin, coord_crt, lat_vecs, sub_lat_vecs):
    "Enter the coordinates of the atomic site in the unit cell and the origin of the lattice (sublattice 1)."
    "Return the sublattice that the input coordinate belongs to."
    "Only works for triangle lattice. The first lattice vector is the horizontal one, the second is the non-horizontal one."

    # Shift the coordinate to the origin
    rel_coord = np.array(coord_crt) - np.array(origin)

    n2 = rel_coord[1] // lat_vecs[1][1]
    n1 = (rel_coord[0] - n2 * lat_vecs[1][0]) // lat_vecs[0][0]

    frac = rel_coord - n1 * lat_vecs[0] - n2 * lat_vecs[1]

    if np.allclose(frac, sub_lat_vecs[0], atol=1e-1):
        return 0
    elif np.allclose(frac, sub_lat_vecs[1], atol=1):
        return 1
    elif np.allclose(frac, sub_lat_vecs[2], atol=1):
        return 2
    else:
        return -1
    

def generate_k_space_sweep(pt_list, num_points):
    """
    Generate a sweep of k-points along the specified path in reciprocal space.

    Parameters:
    pt_list (np.array): List of points in reciprocal space to define the path.
    num_points (int): Total number of k-points to generate along the path.

    Returns:
    b (np.array): 1D array that contains the cumulative distance along the path for each k-point.
    kx_sweep (np.array): 1D array of kx coordinates for the sweep.
    ky_sweep (np.array): 1D array of ky coordinates for the sweep.
    """

    # First calculate the total length of the path in reciprocal space
    total_length = 0
    for i in range(len(pt_list) - 1):
        total_length += np.linalg.norm(pt_list[i+1] - pt_list[i])

    kx_sweep = np.zeros(0)
    ky_sweep = np.zeros(0)
    b = np.zeros(0)
    
    # Now generate the k-points along the path
    for i in range(len(pt_list) - 1):
        start_pt = pt_list[i]
        end_pt = pt_list[i+1]
        segment_length = np.linalg.norm(end_pt - start_pt)
        

        # Generate k-points for this segment
        if i == len(pt_list) - 2:  # Ensure the last point is included in the final segment
            num_points_segment = int(np.ceil(num_points * (segment_length / total_length)))
            kx_segment = np.linspace(start_pt[0], end_pt[0], num_points_segment, endpoint=True)
            ky_segment = np.linspace(start_pt[1], end_pt[1], num_points_segment, endpoint=True)
            b_segment = np.linspace(total_length * i / (len(pt_list) - 1), total_length * (i + 1) / (len(pt_list) - 1), num_points_segment, endpoint=True)
        else:
            num_points_segment = int(np.round(num_points * (segment_length / total_length)))
            kx_segment = np.linspace(start_pt[0], end_pt[0], num_points_segment, endpoint=False)
            ky_segment = np.linspace(start_pt[1], end_pt[1], num_points_segment, endpoint=False)
            b_segment = np.linspace(total_length * i / (len(pt_list) - 1), total_length * (i + 1) / (len(pt_list) - 1), num_points_segment, endpoint=False)

        b = np.hstack((b, b_segment))
        kx_sweep = np.hstack((kx_sweep, kx_segment))
        ky_sweep = np.hstack((ky_sweep, ky_segment))


    return b, kx_sweep, ky_sweep


def visualize_sweep(k_sweep, BZ_points):
    """
    Plot the k-space sweep path along with the first Brillouin zone points.
    """

    plt.figure(figsize=(6, 4))
    plt.axis('equal')
    plt.plot(BZ_points[:, 0], BZ_points[:, 1], '-', label='BZ Points')
    plt.plot(k_sweep[0], k_sweep[1], 'o', label='k-space Sweep')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.title('k-space Sweep Path')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()

    return 


def rotate_points(points, angles):
    """
    Rotate one point in 2D by an array of angles.

    Parameters:
    points (np.array): 1D array of length 2 containing the x and y coordinates of the point to be rotated. 
    angles (np.array): 1D array of rotation angles in radians.

    Returns:
    np.array: 2D array of shape (len(angles), 2) containing the rotated coordinates for each angle.
    """

    # Create the rotation matrix for each angle
    cos_angle = np.cos(angles)
    sin_angle = np.sin(angles)
    rotation_matrices = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]]).transpose(2, 0, 1)

    # Rotate the point using the rotation matrices
    rotated_points = np.einsum('ijk,j->ik', rotation_matrices, points)

    return rotated_points

def mirror_point(pt_coord, line_slope):
    x0 = pt_coord[0]
    y0 = pt_coord[1]

    # first find the angle between the point and the line
    angle = np.arctan2(line_slope, 1) - np.arctan2(y0, x0)

    # then find the mirrored point by rotating (x0, y0) by 2 times the angle
    x1 = x0 * np.cos(2*angle) - y0 * np.sin(2*angle)
    y1 = x0 * np.sin(2*angle) + y0 * np.cos(2*angle)

    return x1, y1



def symmetry_k_points(kx, ky, angles):
    """Return 6 rotated and 6 mirrored-then-rotated (kx, ky) pairs."""
    kx_mirr, ky_mirr = mirror_point((kx, ky), 1 / np.sqrt(3))
    c = np.cos(angles)
    s = np.sin(angles)

    kx_rot = kx * c - ky * s
    ky_rot = kx * s + ky * c
    kx_mrot = kx_mirr * c - ky_mirr * s
    ky_mrot = kx_mirr * s + ky_mirr * c

    return np.concatenate([kx_rot, kx_mrot]), np.concatenate([ky_rot, ky_mrot])



def fourier_symmetrized_amplitude(points, kx_ops, ky_ops):
    """Mean |FT| over all symmetry-related k-points for one sublattice."""

    pts = np.asarray(points)
    x = pts[:, 0]
    y = pts[:, 1]
    amp = pts[:, 2]
    phase0 = np.deg2rad(pts[:, 3])

    phases = np.outer(kx_ops, x) + np.outer(ky_ops, y) + phase0
    ft_vals = np.abs((amp[None, :] * np.exp(-1j * phases)).sum(axis=1))
    return ft_vals.mean()

def fourier_symmetrized_amplitude_squared(points, kx_ops, ky_ops):
    """Mean |FT|^2 over all symmetry-related k-points for one sublattice."""

    pts = np.asarray(points)
    x = pts[:, 0]
    y = pts[:, 1]
    amp = pts[:, 2]
    phase0 = np.deg2rad(pts[:, 3])

    phases = np.outer(kx_ops, x) + np.outer(ky_ops, y) + phase0
    ft_vals = np.abs((amp[None, :] * np.exp(-1j * phases)).sum(axis=1)) ** 2
    return ft_vals.mean()

####### Visualization functions
    

def real_space_plot(
    x_coords,
    y_coords,
    value,
    max_amp,
    radius,
    out_path=None,
    xlim=(-75, 75),
    ylim=(-75, 75),
    cmap="bwr",
    cbar_ticks=(-3, 0, 3),
    cbar_label=r'$p(\mathbf{r}, f)$ (Pa)',
    figsize=(2.5, 2.5),
    dpi=300,
    cbar_shrink=0.5,
):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.gca()
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    plt.yticks([])
    plt.xticks([])

    norm = colors.Normalize(vmin=-max_amp, vmax=max_amp)
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        shrink=cbar_shrink,
    )

    for x_i, y_i, v_i in zip(x_coords, y_coords, value):
        circle = ptc.Ellipse(
            (x_i, y_i),
            width=radius * 2,
            height=radius * 2,
            edgecolor="black",
            facecolor=plt.get_cmap(cmap)((v_i + max_amp) / (2 * max_amp)),
            fill=True,
            alpha=1,
            linewidth=0.25,
        )
        ax.add_patch(circle)

    cbar_ticks = np.array(cbar_ticks)
    cbar.set_ticks(cbar_ticks)
    cbar.outline.set_linewidth(0.25)
    cbar.set_label(cbar_label, fontsize=6, rotation=270, labelpad=5)
    cbar.ax.set_yticklabels([f"{tick:g}" for tick in cbar_ticks], fontsize=6)
    cbar.ax.tick_params(width=0.25, length=1)

    if out_path is not None:
        plt.savefig(out_path, bbox_inches="tight")

    return fig, ax


def plot_2d_fourier_transform(
    reciprocal_data,
    kxs,
    kys,
    BZ_points,
    out_dir,
    freq,
    scale=1,
    cmap=stmpy.cm.blue3_r,
    vmin=0,
    vmax=1,
    figsize=(3, 3),
    save=True,
    save_fn='2D_DFT',
):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        reciprocal_data / scale,
        extent=[kxs[0], kxs[-1], kys[0], kys[-1]],
        aspect="equal",
        origin="lower",
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
    )
    ax.plot(BZ_points[:, 0], BZ_points[:, 1], "r:", linewidth=0.25)

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.25)
    ax.set_xlim([kxs[0], kxs[-1]])
    ax.set_ylim([kys[0], kys[-1]])
    ax.set_xticks([-400, 0, 400], [400, 0, 400], size=8)
    ax.set_yticks([-400, 0, 400], [400, 0, 400], size=8)
    ax.xaxis.set_tick_params(width=0.25, length=1)
    ax.yaxis.set_tick_params(width=0.25, length=1)
    ax.set_xlabel(r"$k_x$ (1/m)", fontsize=8)
    ax.set_ylabel(r"$k_y$ (1/m)", fontsize=8)

    cb = plt.colorbar(im, location="right", pad=0.07, aspect=15, shrink=0.7)
    cb.set_ticks([0, 1])
    cb.ax.yaxis.set_tick_params(width=0.25, length=1)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.25)
    cb.set_label(r"$|\tilde{p}(\mathbf{k}, f)|$ (A.U.)", fontsize=8, rotation=270, labelpad=5)

    if save:
        fig.savefig(out_dir + "/{}_{}Hz.pdf".format(save_fn, freq), bbox_inches="tight")

    return fig, ax, cb