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


def compute_berry_curvature_vectorized(KX, KY, KZ, M):
    """
    Compute Berry Curvature for a 2-band Weyl Semimetal Hamiltonian (vectorized).
    
    Hamiltonian: H(k) = sin(kx)*σx + sin(ky)*σy + (M - cos(kx) - cos(ky) - cos(kz))*σz
    
    The d-vector is: d = (sin(kx), sin(ky), M - cos(kx) - cos(ky) - cos(kz))
    
    Berry curvature for 2-band system:
    Ω_i = (1/2) * ε_ijk * d̂ · (∂d̂/∂k_j × ∂d̂/∂k_k)
        = (1/2|d|³) * d · (∂d/∂k_j × ∂d/∂k_k)
    
    Returns: (Energy, Berry_Fx, Berry_Fy, Berry_Fz)
    """
    # d-vector components
    dx = np.sin(KX)
    dy = np.sin(KY)
    dz = M - np.cos(KX) - np.cos(KY) - np.cos(KZ)
    
    # Energy eigenvalues: E = |d| = sqrt(dx² + dy² + dz²)
    E = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Avoid division by zero at Weyl points
    dist_cubed = np.maximum(E**3, 1e-10)
    
    # Gradient components ∂d_i/∂k_j:
    # ∂d/∂kx = (cos(kx), 0, sin(kx))
    # ∂d/∂ky = (0, cos(ky), sin(ky))
    # ∂d/∂kz = (0, 0, sin(kz))
    
    ddx_dkx = np.cos(KX)  # ∂dx/∂kx
    ddy_dky = np.cos(KY)  # ∂dy/∂ky
    ddz_dkx = np.sin(KX)  # ∂dz/∂kx
    ddz_dky = np.sin(KY)  # ∂dz/∂ky
    ddz_dkz = np.sin(KZ)  # ∂dz/∂kz
    
    # Berry curvature components using Ω_i = d · (∂d/∂k_j × ∂d/∂k_k) / (2|d|³)
    
    # Ω_x = d · (∂d/∂ky × ∂d/∂kz) / (2|d|³)
    # ∂d/∂ky = (0, cos(ky), sin(ky))
    # ∂d/∂kz = (0, 0, sin(kz))
    # ∂d/∂ky × ∂d/∂kz = (cos(ky)*sin(kz) - sin(ky)*0,
    #                     sin(ky)*0 - 0*sin(kz),
    #                     0*0 - cos(ky)*0)
    #                  = (cos(ky)*sin(kz), 0, 0)
    cross_yz_x = ddy_dky * ddz_dkz  # cos(ky)*sin(kz)
    cross_yz_y = np.zeros_like(KX)
    cross_yz_z = np.zeros_like(KX)
    Berry_Fx = (dx * cross_yz_x + dy * cross_yz_y + dz * cross_yz_z) / (2 * dist_cubed)
    
    # Ω_y = d · (∂d/∂kz × ∂d/∂kx) / (2|d|³)
    # ∂d/∂kz = (0, 0, sin(kz))
    # ∂d/∂kx = (cos(kx), 0, sin(kx))
    # ∂d/∂kz × ∂d/∂kx = (0*sin(kx) - sin(kz)*0,
    #                     sin(kz)*cos(kx) - 0*sin(kx),
    #                     0*0 - 0*cos(kx))
    #                  = (0, sin(kz)*cos(kx), 0)
    cross_zx_x = np.zeros_like(KX)
    cross_zx_y = ddz_dkz * ddx_dkx  # sin(kz)*cos(kx)
    cross_zx_z = np.zeros_like(KX)
    Berry_Fy = (dx * cross_zx_x + dy * cross_zx_y + dz * cross_zx_z) / (2 * dist_cubed)
    
    # Ω_z = d · (∂d/∂kx × ∂d/∂ky) / (2|d|³)
    # ∂d/∂kx = (cos(kx), 0, sin(kx))
    # ∂d/∂ky = (0, cos(ky), sin(ky))
    # ∂d/∂kx × ∂d/∂ky = (0*sin(ky) - sin(kx)*cos(ky),
    #                     sin(kx)*0 - cos(kx)*sin(ky),
    #                     cos(kx)*cos(ky) - 0*0)
    #                  = (-sin(kx)*cos(ky), -cos(kx)*sin(ky), cos(kx)*cos(ky))
    cross_xy_x = -ddz_dkx * ddy_dky  # -sin(kx)*cos(ky)
    cross_xy_y = -ddx_dkx * ddz_dky  # -cos(kx)*sin(ky)
    cross_xy_z = ddx_dkx * ddy_dky   # cos(kx)*cos(ky)
    Berry_Fz = (dx * cross_xy_x + dy * cross_xy_y + dz * cross_xy_z) / (2 * dist_cubed)
    
    return E, Berry_Fx, Berry_Fy, Berry_Fz


def find_weyl_points(M):
    """
    Find approximate Weyl point locations for given mass parameter M.
    
    Weyl points occur where d = 0, i.e., where:
    - sin(kx) = 0  →  kx = 0, ±π
    - sin(ky) = 0  →  ky = 0, ±π
    - M - cos(kx) - cos(ky) - cos(kz) = 0
    
    For kx = ky = 0: M - 2 - cos(kz) = 0  →  cos(kz) = M - 2
    Valid when |M - 2| ≤ 1, i.e., 1 ≤ M ≤ 3
    """
    weyl_points = []
    
    # Check kx = ky = 0 case
    if abs(M - 2) <= 1:
        kz_weyl = np.arccos(M - 2)
        weyl_points.append((0, 0, kz_weyl))
        weyl_points.append((0, 0, -kz_weyl))
    
    # Other high-symmetry points can be checked similarly
    # kx = π, ky = 0: M - (-1) - 1 - cos(kz) = 0 → cos(kz) = M
    if abs(M) <= 1:
        kz_weyl = np.arccos(M)
        weyl_points.append((np.pi, 0, kz_weyl))
        weyl_points.append((np.pi, 0, -kz_weyl))
        weyl_points.append((0, np.pi, kz_weyl))
        weyl_points.append((0, np.pi, -kz_weyl))
    
    # kx = π, ky = π: M - (-1) - (-1) - cos(kz) = 0 → cos(kz) = M + 2
    if abs(M + 2) <= 1:
        kz_weyl = np.arccos(M + 2)
        weyl_points.append((np.pi, np.pi, kz_weyl))
        weyl_points.append((np.pi, np.pi, -kz_weyl))
    
    return weyl_points


# Generate data for each mass parameter value
for frame_idx, M in enumerate(M_values):
    print(f"Frame {frame_idx+1}/{len(M_values)}: M = {M:.3f}")
    
    # Find Weyl points for this M value
    weyl_pts = find_weyl_points(M)
    if weyl_pts:
        print(f"  Weyl points at: {[(f'({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})') for p in weyl_pts[:4]]}")
    
    # Compute Berry Curvature (fully vectorized - no loops!)
    Energy, Berry_Fx, Berry_Fy, Berry_Fz = compute_berry_curvature_vectorized(KX, KY, KZ, M)
    
    # Compute magnitude of Berry Curvature for volume rendering
    Berry_Magnitude = np.sqrt(Berry_Fx**2 + Berry_Fy**2 + Berry_Fz**2)
    
    # Compute divergence of Berry curvature (should be zero except at Weyl points)
    # This acts as a "Weyl point detector"
    dk = k_vals[1] - k_vals[0]
    div_Berry = np.zeros_like(Berry_Magnitude)
    div_Berry[1:-1, :, :] += (Berry_Fx[2:, :, :] - Berry_Fx[:-2, :, :]) / (2 * dk)
    div_Berry[:, 1:-1, :] += (Berry_Fy[:, 2:, :] - Berry_Fy[:, :-2, :]) / (2 * dk)
    div_Berry[:, :, 1:-1] += (Berry_Fz[:, :, 2:] - Berry_Fz[:, :, :-2]) / (2 * dk)
    
    # Chern number indicator (integrated Berry curvature on slices)
    # Positive/negative values indicate monopole charges
    Monopole_Charge = div_Berry / (2 * np.pi)
    
    # Log-scale magnitude for better visualization (Berry curvature diverges at Weyl points)
    Berry_Magnitude_Log = np.log10(Berry_Magnitude + 1e-10)
    
    # Ensure arrays are contiguous and in Fortran order for VTK
    Energy = np.ascontiguousarray(Energy)
    Berry_Fx = np.ascontiguousarray(Berry_Fx)
    Berry_Fy = np.ascontiguousarray(Berry_Fy)
    Berry_Fz = np.ascontiguousarray(Berry_Fz)
    Berry_Magnitude = np.ascontiguousarray(Berry_Magnitude)
    Berry_Magnitude_Log = np.ascontiguousarray(Berry_Magnitude_Log)
    Monopole_Charge = np.ascontiguousarray(Monopole_Charge)
    
    # Save to VTK file
    filename = os.path.join(output_dir, f"weyl_M_{frame_idx:03d}")
    gridToVTK(
        filename, 
        np.ascontiguousarray(k_vals), 
        np.ascontiguousarray(k_vals), 
        np.ascontiguousarray(k_vals),
        pointData={
            "Energy": Energy,
            "BerryCurvature": (Berry_Fx, Berry_Fy, Berry_Fz),
            "BerryMagnitude": Berry_Magnitude,
            "BerryMagnitudeLog": Berry_Magnitude_Log,
            "MonopoleCharge": Monopole_Charge,
        }
    )
    
    # Print some statistics
    print(f"  Energy range: [{Energy.min():.4f}, {Energy.max():.4f}]")
    print(f"  Berry magnitude range: [{Berry_Magnitude.min():.4e}, {Berry_Magnitude.max():.4e}]")
    print(f"  Total monopole charge: {Monopole_Charge.sum() * dk**3:.4f}")
    print(f"  Saved: {filename}.vtr")
    print()

print("=" * 60)
print(f"Done! Generated {len(M_values)} frames in '{output_dir}' directory.")
print("=" * 60)
print()
print("ParaView Visualization Tips:")
print("-" * 60)
print("1. Open the .vtr files as a time series in ParaView")
print("2. For Berry curvature field lines:")
print("   - Use 'Stream Tracer' filter on BerryCurvature vector field")
print("   - Seed points near Weyl points (where Energy → 0)")
print("3. For Weyl point visualization:")
print("   - Use 'Contour' filter on Energy with value ~0.01")
print("   - Or use 'Volume Rendering' on BerryMagnitudeLog")
print("4. For monopole charges:")
print("   - Use 'Threshold' on MonopoleCharge to show ±values")
print("   - Red/blue coloring shows +/- chirality")
print("5. Animate through M values to see Weyl point creation/annihilation")