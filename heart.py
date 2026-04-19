import numpy as np
import pyvista as pv
from skimage import measure

# --- SECTION 1: Space Definition ---
# Increase N for higher resolution heart surface
N = 350
x = np.linspace(-1.5, 1.5, N)
y = np.linspace(-1.5, 1.5, N)
z = np.linspace(-1.5, 1.5, N)

# Create the 3D grid using indexing='ij' to preserve axes (X, Y, Z)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# --- SECTION 2: Equation Evaluation ---
# Evaluate Taubin's heart algebraic surface equation
F = (X**2 + (9/4)*Y**2 + Z**2 - 1)**3 - (X**2)*(Z**3) - (9/80)*(Y**2)*(Z**3)

print("Calculating the isosurface and generating polygons... (please wait a few seconds)")

# --- SECTION 3: Polygon Generation ---
# Extract vertices and faces using Marching Cubes
verts, faces, normals, values = measure.marching_cubes(F, level=0.0)

# PyVista formatting: prepend '3' to each triangular face definition
faces_pv = np.column_stack((np.full(len(faces), 3), faces)).flatten()

# --- SECTION 4: Mesh Creation & Optimization ---
heart_mesh = pv.PolyData(verts, faces_pv)
heart_mesh.compute_normals(inplace=True) 

print(f"Done! The mesh has been generated:")
print(f" -> Polygons (Faces): {heart_mesh.n_cells:,}")
print(f" -> Vertices: {heart_mesh.n_points:,}")

# --- SECTION 5: Creating and Position the Standing Text Inside ---
print("Creating text and aligning inside...")
anel_text = pv.Text3D("Anel", depth=0.2)

# 1. ROTATE: Rotate text 90 degrees around X-axis
anel_text.rotate_x(90, inplace=True)

# 2. LOCAL ALIGNMENT: Align the text's local geometric center to (0,0,0)
text_center = anel_text.center
anel_text.translate([-text_center[0], -text_center[1], -text_center[2]], inplace=True)

# 3. GET HEART GEOMETRY BOUNDS
bounds = heart_mesh.bounds
heart_height = bounds[5] - bounds[4] # Z

# 4. SCALE TEXT: Adjusting size to be much smaller (one fourth)
text_height = anel_text.bounds[5] - anel_text.bounds[4]
final_scale_factor = (heart_height * 0.25) / text_height
anel_text.scale(final_scale_factor, inplace=True)

# 5. POSITIONING: Move text to heart's geometric center
heart_center = heart_mesh.center

# 6. VERTICAL OFFSET (Z)
z_offset = heart_height * 0.12 
anel_text.translate([heart_center[0], heart_center[1], heart_center[2] + z_offset], inplace=True)

# --- SECTION 6: Advanced Rendering & Cinematic Improvements ---
plotter = pv.Plotter()

# 6.1 Adding the Heart Mesh (The "Ruby" look)
plotter.add_mesh(
    heart_mesh,
    color='#800020',       # Darker crimson / Ruby red
    smooth_shading=True,   
    specular=1.0,          # Increased for a glowing gemstone look
    opacity=0.3,           # Lower opacity to clearly see the gold text
    ambient=0.1            
)

# 6.2 Adding the Text Mesh (Warm Gold)
plotter.add_mesh(
    anel_text,
    color='#FFD700',       # Gold contrast against the deep red
    smooth_shading=True,
    specular=0.8,
    ambient=0.2
)

# 6.3 Add a 2D Poetic Dedication
plotter.add_text(
    "Un abrazo que dure para siempre.\nAnel, te quiero.", 
    position='lower_right', 
    color='gray', 
    font_size=12,
    font='courier'
)

# 6.4 Scene background and Professional Lighting
plotter.set_background('#121212') # Dark background
plotter.enable_3_lights()         # Studio 3-point lighting setupplotter.set_background('#121212') # Dark background


# 6.5 Elegant Spin Animation
def rotate_camera(step):
    plotter.camera.azimuth += 0.5 # Adjust speed here (higher is faster)

# Triggers the rotation every 50ms automatically
plotter.add_timer_event(max_steps=100000, duration=50, callback=rotate_camera)

print("Opening interaction window... The heart will now spin elegantly.")
# Show the window
plotter.show()