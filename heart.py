import numpy as np
import pyvista as pv
from skimage import measure

# --- SECTION 1: Space Definition ---
# Increase N for higher resolution heart surface (more polygons)
# 350 creates ~1.2 million polygons. 500 requires substantial RAM.
N = 350
x = np.linspace(-1.5, 1.5, N)
y = np.linspace(-1.5, 1.5, N)
z = np.linspace(-1.5, 1.5, N)

# Create the 3D grid using indexing='ij' to preserve axes (X, Y, Z)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# --- SECTION 2: Equation Evaluation ---
# Evaluate Taubin's heart algebraic surface equation
# F = 0 describes the geometry
F = (X**2 + (9/4)*Y**2 + Z**2 - 1)**3 - (X**2)*(Z**3) - (9/80)*(Y**2)*(Z**3)

print("Calculating the isosurface and generating polygons... (please wait a few seconds)")

# --- SECTION 3: Polygon Generation ---
# Extract vertices and faces using Marching Cubes (isosurface at level 0.0)
verts, faces, normals, values = measure.marching_cubes(F, level=0.0)

# PyVista formatting: prepend '3' to each triangular face definition
faces_pv = np.column_stack((np.full(len(faces), 3), faces)).flatten()

# --- SECTION 4: Mesh Creation & Optimization ---
heart_mesh = pv.PolyData(verts, faces_pv)
# Calculate normals for proper lighting on the high-polygon mesh
heart_mesh.compute_normals(inplace=True) 

print(f"Done! The mesh has been generated:")
print(f" -> Polygons (Faces): {heart_mesh.n_cells:,}")
print(f" -> Vertices: {heart_mesh.n_points:,}")


# --- SECTION 5: Creating and Position the Standing Text Inside ---
print("Creating text and aligning inside...")
# Initialize 3D text. Depth adds thickness to the letters.
anel_text = pv.Text3D("Anel", depth=0.2)

# 1. ROTATE: Rotate text 90 degrees around X-axis to make it stand up vertically
anel_text.rotate_x(90, inplace=True)

# 2. LOCAL ALIGNMENT: Align the text's local geometric center to (0,0,0)
text_center = anel_text.center
anel_text.translate([-text_center[0], -text_center[1], -text_center[2]], inplace=True)

# 3. GET HEART GEOMETRY BOUNDS:
# bounds format: [xmin, xmax, ymin, ymax, zmin, zmax]
bounds = heart_mesh.bounds
# heart_width = bounds[1] - bounds[0] # X
# heart_depth = bounds[3] - bounds[2] # Y
heart_height = bounds[5] - bounds[4] # Z

# 4. SCALE TEXT: Adjusting size to be much smaller.
text_height = anel_text.bounds[5] - anel_text.bounds[4]

# FIX APPLIED HERE:
# Instead of targeting 85% (0.85) of heart height, we target one fourth (0.25).
# original_size_factor = (heart_height * 0.85) / text_height
final_scale_factor = (heart_height * 0.25) / text_height

anel_text.scale(final_scale_factor, inplace=True)

# 5. POSITIONING: Move text to heart's geometric center
heart_center = heart_mesh.center

# 6. VERTICAL OFFSET (Z): The center of Taubin's heart is not its visual center.
# We keep the same offset from before (12% of height) to shift the smaller text up.
z_offset = heart_height * 0.12 
anel_text.translate([heart_center[0], heart_center[1], heart_center[2] + z_offset], inplace=True)


# --- SECTION 6: Advanced Rendering ---
plotter = pv.Plotter()

# 6.1 Adding the Heart Mesh (Translucent)
plotter.add_mesh(
    heart_mesh,
    color='#e30022',       # Intense red
    smooth_shading=True,   # Essential for smooth look of millions of polygons
    specular=0.8,          # High specular reflection for a glossy look
    opacity=0.45,          # KEY: Set opacity < 1.0 to see inside!
    ambient=0.1            # Base shadow
)

# 6.2 Adding the Text Mesh (Solid white inside)
plotter.add_mesh(
    anel_text,
    color='white',         # High contrast against red
    smooth_shading=True,
    specular=0.5,
    ambient=0.1
)

# Scene background and lighting
plotter.set_background('#121212') # Dark background
plotter.add_light(pv.Light(position=(2, 2, 2), focal_point=(0, 0, 0), color='white'))

print("Opening interaction window... Rotate and zoom with the mouse.")
# Show the window (you can rotate and zoom with the mouse)
plotter.show()