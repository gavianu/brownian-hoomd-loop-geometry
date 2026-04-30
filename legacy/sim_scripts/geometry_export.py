"""
Export Geometry to GSD

This script builds the sealed loop geometry and exports it to a GSD file for visualization.
No particles (tracers), no simulation steps.
Designed to run on GPU if available.

Device structure:
  [IN CUBE]----[5 FUNNELS with varying radii]----[OUT CUBE]
      ^                                              |
      |_____________[RETURN TUBE]____________________|

- Uses GPU device for initialization (though no dynamics run).
"""

import os
import numpy as np
import hoomd

# ============= CONFIG =============
class CFG:
    # Box
    Lx, Ly, Lz = 450.0, 350.0, 300.0
    
    # === CHAMBERS ===
    CUBE_SIZE = 80.0
    CUBE_Y = 80.0
    X_IN  = -140.0
    X_OUT =  140.0
    
    # === FUNNELS ===
    N_FUNNELS = 5
    FUNNEL_RADII = [15.0, 12.0, 9.0, 6.0, 3.0]  # Different diameters, descending
    FUNNEL_SPACING_Z = 35.0
    FUNNEL_Y = 80.0
    
    # === RETURN TUBE ===
    RETURN_Y = -80.0
    RETURN_RADIUS = 25.0
    
    # === VERTICAL CONNECTORS (cube bottom → return tube) ===
    CONNECTOR_RADIUS = 20.0
    
    # === WALL BEADS ===
    WALL_SPACING = 2.0  # Smaller for denser walls, better sealing

# ============= GEOMETRY =============

def make_hollow_cylinder(x1, x2, y_c, z_c, radius, spacing):
    """Cylinder surface along X axis"""
    pts = []
    length = abs(x2 - x1)
    
    n_circ = max(int(2 * np.pi * radius / spacing), 16)
    n_axial = max(int(length / spacing) + 1, 5)
    
    x_vals = np.linspace(min(x1, x2), max(x1, x2), n_axial)
    thetas = np.linspace(0, 2*np.pi, n_circ, endpoint=False)
    
    for x in x_vals:
        for theta in thetas:
            y = y_c + radius * np.cos(theta)
            z = z_c + radius * np.sin(theta)
            pts.append([x, y, z])
    
    return np.array(pts)

def make_hollow_cylinder_vertical(y1, y2, x_c, z_c, radius, spacing):
    """Cylinder surface along Y axis (vertical)"""
    pts = []
    length = abs(y2 - y1)
    
    n_circ = max(int(2 * np.pi * radius / spacing), 16)
    n_axial = max(int(length / spacing) + 1, 5)
    
    y_vals = np.linspace(min(y1, y2), max(y1, y2), n_axial)
    thetas = np.linspace(0, 2*np.pi, n_circ, endpoint=False)
    
    for y in y_vals:
        for theta in thetas:
            x = x_c + radius * np.cos(theta)
            z = z_c + radius * np.sin(theta)
            pts.append([x, y, z])
    
    return np.array(pts)

def make_cube_shell(center, size, spacing):
    """Cube with ALL 6 faces (we'll remove connection zones later)"""
    cx, cy, cz = center
    s2 = size / 2
    pts = []
    
    n = int(size / spacing) + 1
    grid = np.linspace(-s2, s2, n)
    
    # All 6 faces
    for y in grid:
        for z in grid:
            pts.append([cx + s2, cy + y, cz + z])  # +X
            pts.append([cx - s2, cy + y, cz + z])  # -X
    
    for x in grid:
        for z in grid:
            pts.append([cx + x, cy + s2, cz + z])  # +Y
            pts.append([cx + x, cy - s2, cz + z])  # -Y
    
    for x in grid:
        for y in grid:
            pts.append([cx + x, cy + y, cz + s2])  # +Z
            pts.append([cx + x, cy + y, cz - s2])  # -Z
    
    return np.array(pts)

def remove_beads_in_cylinder(points, x1, x2, y_c, z_c, radius):
    """Remove points inside a cylindrical region along X"""
    keep = []
    x_min, x_max = min(x1, x2), max(x1, x2)
    
    for p in points:
        x, y, z = p
        if x_min <= x <= x_max:
            dist = np.sqrt((y - y_c)**2 + (z - z_c)**2)
            if dist <= radius * 1.1:  # Slightly larger to ensure clean hole
                continue
        keep.append(p)
    
    return np.array(keep) if keep else np.empty((0, 3))

def remove_beads_in_cylinder_vertical(points, y1, y2, x_c, z_c, radius):
    """Remove points inside a vertical cylindrical region along Y"""
    keep = []
    y_min, y_max = min(y1, y2), max(y1, y2)
    
    for p in points:
        x, y, z = p
        if y_min <= y <= y_max:
            dist = np.sqrt((x - x_c)**2 + (z - z_c)**2)
            if dist <= radius * 1.1:
                continue
        keep.append(p)
    
    return np.array(keep) if keep else np.empty((0, 3))

def build_device(cfg):
    """Build complete sealed device with internal circulation"""
    
    print("\n=== Building Device ===")
    
    # Calculate funnel Z positions
    z_funnels = np.linspace(
        -cfg.FUNNEL_SPACING_Z * (cfg.N_FUNNELS - 1) / 2,
        cfg.FUNNEL_SPACING_Z * (cfg.N_FUNNELS - 1) / 2,
        cfg.N_FUNNELS
    )
    
    # === 1. IN CUBE (with holes for funnels and vertical connector) ===
    print("  Building IN cube...")
    cube_in = make_cube_shell((cfg.X_IN, cfg.CUBE_Y, 0), cfg.CUBE_SIZE, cfg.WALL_SPACING)
    
    # Remove beads where funnels connect (+X face)
    for i, z_f in enumerate(z_funnels):
        r = cfg.FUNNEL_RADII[i]
        cube_in = remove_beads_in_cylinder(
            cube_in,
            cfg.X_IN + cfg.CUBE_SIZE/2 - 2, cfg.X_IN + cfg.CUBE_SIZE/2 + 2,
            cfg.FUNNEL_Y, z_f,
            r * 1.2
        )
    
    # Remove beads where vertical connector attaches (bottom)
    cube_in = remove_beads_in_cylinder_vertical(
        cube_in,
        cfg.CUBE_Y - cfg.CUBE_SIZE/2 - 2, cfg.CUBE_Y - cfg.CUBE_SIZE/2 + 2,
        cfg.X_IN, 0,
        cfg.CONNECTOR_RADIUS * 1.2
    )
    
    print(f"    IN cube: {len(cube_in)} beads")
    
    # === 2. OUT CUBE (with holes for funnels and vertical connector) ===
    print("  Building OUT cube...")
    cube_out = make_cube_shell((cfg.X_OUT, cfg.CUBE_Y, 0), cfg.CUBE_SIZE, cfg.WALL_SPACING)
    
    # Remove beads where funnels connect (-X face)
    for i, z_f in enumerate(z_funnels):
        r = cfg.FUNNEL_RADII[i]
        cube_out = remove_beads_in_cylinder(
            cube_out,
            cfg.X_OUT - cfg.CUBE_SIZE/2 - 2, cfg.X_OUT - cfg.CUBE_SIZE/2 + 2,
            cfg.FUNNEL_Y, z_f,
            r * 1.2
        )
    
    # Remove beads where vertical connector attaches (bottom)
    cube_out = remove_beads_in_cylinder_vertical(
        cube_out,
        cfg.CUBE_Y - cfg.CUBE_SIZE/2 - 2, cfg.CUBE_Y - cfg.CUBE_SIZE/2 + 2,
        cfg.X_OUT, 0,
        cfg.CONNECTOR_RADIUS * 1.2
    )
    
    print(f"    OUT cube: {len(cube_out)} beads")
    
    # === 3. FUNNELS (horizontal cylinders with varying radii) ===
    print("  Building funnels...")
    funnels = []
    x_start = cfg.X_IN + cfg.CUBE_SIZE / 2
    x_end = cfg.X_OUT - cfg.CUBE_SIZE / 2
    
    for i, z_f in enumerate(z_funnels):
        r = cfg.FUNNEL_RADII[i]
        funnel = make_hollow_cylinder(
            x_start, x_end,
            cfg.FUNNEL_Y, z_f,
            r,
            cfg.WALL_SPACING
        )
        funnels.append(funnel)
        print(f"    Funnel {i+1}: {len(funnel)} beads at z={z_f:.1f}, r={r}")
    
    # === 4. VERTICAL CONNECTORS (cube bottoms → return tube) ===
    print("  Building vertical connectors...")
    
    # IN connector
    y_cube_bottom = cfg.CUBE_Y - cfg.CUBE_SIZE / 2
    y_return_top = cfg.RETURN_Y + cfg.RETURN_RADIUS
    
    connector_in = make_hollow_cylinder_vertical(
        y_return_top, y_cube_bottom,
        cfg.X_IN, 0,
        cfg.CONNECTOR_RADIUS,
        cfg.WALL_SPACING
    )
    print(f"    IN connector: {len(connector_in)} beads")
    
    # OUT connector
    connector_out = make_hollow_cylinder_vertical(
        y_return_top, y_cube_bottom,
        cfg.X_OUT, 0,
        cfg.CONNECTOR_RADIUS,
        cfg.WALL_SPACING
    )
    print(f"    OUT connector: {len(connector_out)} beads")
    
    # === 5. RETURN TUBE (horizontal, bottom) ===
    print("  Building return tube...")
    return_tube = make_hollow_cylinder(
        cfg.X_IN, cfg.X_OUT,  # From IN to OUT for consistency
        cfg.RETURN_Y, 0,
        cfg.RETURN_RADIUS,
        cfg.WALL_SPACING
    )
    print(f"    Return tube: {len(return_tube)} beads")
    
    # === COMBINE ALL ===
    all_parts = [cube_in, cube_out, connector_in, connector_out, return_tube] + funnels
    all_parts = [p for p in all_parts if len(p) > 0]
    
    geometry = np.vstack(all_parts).astype(np.float32)
    print(f"\n  TOTAL WALLS: {len(geometry)} beads\n")
    
    return geometry

# ============= MAIN =============

def main():
    os.makedirs('sim/out_loop', exist_ok=True)
    
    print("🖥️  Using GPU if available\n")
    device = hoomd.device.GPU()  # Use GPU
    sim = hoomd.Simulation(device=device, seed=123)
    
    # Build sealed device geometry (walls only)
    walls = build_device(CFG)
    
    # Create snapshot with only walls
    snap = hoomd.Snapshot()
    snap.configuration.box = [CFG.Lx, CFG.Ly, CFG.Lz, 0, 0, 0]
    snap.particles.N = len(walls)
    snap.particles.types = ['W']
    snap.particles.typeid[:] = 0
    snap.particles.position[:] = walls
    
    # Write to GSD
    gsd_filename = 'sim/out_loop/geometry.gsd'
    hoomd.write.GSD.write(snap, filename=gsd_filename, mode='wb')
    
    print("\n✅ DONE!")
    print(f"📁 {gsd_filename}")
    print("Visualize the geometry in OVITO or similar tool.")

if __name__ == '__main__':
    main()