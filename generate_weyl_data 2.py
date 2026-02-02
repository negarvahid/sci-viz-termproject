import numpy as np
from pyevtk.hl import gridToVTK
import os

# --- CONFIGURATION ---
N = 64              # Grid Size (64^3 for reasonable computation time, increase to 100+ for higher quality)
M_values = np.linspace(1.0, 3.0, 20)  # Mass parameter range for animation
output_dir = "./weyl_output"
os.makedirs(output_dir, exist_ok=True)

# Momentum space grid (Brillouin Zone: -pi to pi)
k_vals = np.linspace(-np.pi, np.pi, N)
KX, KY, KZ = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')

print("=" * 60)
print("Generating Berry Curvature Data for Weyl Semimetals")
print("=" * 60)
print(f"Grid size: {N}^3 = {N**3:,} points")
print(f"Mass parameter range: {M_values[0]:.2f} to {M_values[-1]:.2f}")
print(f"Number of frames: {len(M_values)}")
print()

# Pauli Matrices
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

def compute_berry_curvature(kx, ky, kz, M):
    """
    Compute Berry Curvature for a 2-band Weyl Semimetal Hamiltonian.
    
    Hamiltonian: H(k) = sin(kx)*sx + sin(ky)*sy + (M - cos(kx) - cos(ky) - cos(kz))*sz
    
    Returns: (Energy, Berry_Fx, Berry_Fy, Berry_Fz)
    """
    # d-vector components
    dx = np.sin(kx)
    dy = np.sin(ky)
    dz = M - np.cos(kx) - np.cos(ky) - np.cos(kz)
    
    # Energy eigenvalues: E = ±sqrt(dx^2 + dy^2 + dz^2)
    E = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Avoid division by zero at Weyl points
    dist = E**3
    dist = np.where(dist < 1e-8, 1e-8, dist)
    
    # Analytical Berry Curvature for 2-level system
    # Omega = (d . (partial_i d × partial_j d)) / (2 * E^3)
    
    # Gradients of d-vector
    ddx_dkx = np.cos(kx)
    ddy_dky = np.cos(ky)
    ddz_dkz = np.sin(kz)
    
    # Cross products for Berry Curvature components
    # Omega_x = (d . (partial_y d × partial_z d)) / (2*E^3)
    # partial_y d = (0, cos(ky), sin(ky))
    # partial_z d = (0, 0, sin(kz))
    # Cross = (cos(ky)*sin(kz), 0, 0)
    Berry_Fx = (dx * np.cos(ky) * np.sin(kz)) / (2 * dist)
    
    # Omega_y = (d . (partial_z d × partial_x d)) / (2*E^3)
    # partial_z d = (0, 0, sin(kz))
    # partial_x d = (cos(kx), 0, sin(kx))
    # Cross = (0, -cos(kx)*sin(kz), 0)
    Berry_Fy = (dy * np.cos(kx) * np.sin(kz)) / (2 * dist)
    
    # Omega_z = (d . (partial_x d × partial_y d)) / (2*E^3)
    # partial_x d = (cos(kx), 0, sin(kx))
    # partial_y d = (0, cos(ky), sin(ky))
    # Cross = (0, 0, cos(kx)*cos(ky))
    Berry_Fz = (dz * np.cos(kx) * np.cos(ky)) / (2 * dist)
    
    return E, Berry_Fx, Berry_Fy, Berry_Fz

# Generate data for each mass parameter value
for frame_idx, M in enumerate(M_values):
    print(f"Frame {frame_idx+1}/{len(M_values)}: M = {M:.3f}")
    
    # Initialize arrays
    Energy = np.zeros((N, N, N))
    Berry_Fx = np.zeros((N, N, N))
    Berry_Fy = np.zeros((N, N, N))
    Berry_Fz = np.zeros((N, N, N))
    
    # Compute Berry Curvature at each point
    for i in range(N):
        if (i + 1) % 16 == 0:
            print(f"  Progress: {i+1}/{N} slices")
        for j in range(N):
            for l in range(N):
                kx, ky, kz = k_vals[i], k_vals[j], k_vals[l]
                E, Fx, Fy, Fz = compute_berry_curvature(kx, ky, kz, M)
                Energy[i, j, l] = E
                Berry_Fx[i, j, l] = Fx
                Berry_Fy[i, j, l] = Fy
                Berry_Fz[i, j, l] = Fz
    
    # Compute magnitude of Berry Curvature for volume rendering
    Berry_Magnitude = np.sqrt(Berry_Fx**2 + Berry_Fy**2 + Berry_Fz**2)
    
    # Save to VTK file
    filename = os.path.join(output_dir, f"weyl_M_{frame_idx:03d}")
    gridToVTK(filename, k_vals, k_vals, k_vals,
              pointData={
                  "Energy": Energy,
                  "BerryCurvature": (Berry_Fx, Berry_Fy, Berry_Fz),
                  "BerryMagnitude": Berry_Magnitude
              })
    
    print(f"  Saved: {filename}.vtr")
    print()

print("=" * 60)
print(f"Done! Generated {len(M_values)} frames in '{output_dir}' directory.")
print("=" * 60)


