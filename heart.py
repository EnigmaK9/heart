import numpy as np
import pyvista as pv
from skimage import measure

# 1. Define the space resolution. 
# N=350 generates approximately 1.2 million polygons. 
# You can increase it to N=450 or N=500 if you have enough RAM.
N = 350
x = np.linspace(-1.5, 1.5, N)
y = np.linspace(-1.5, 1.5, N)
z = np.linspace(-1.5, 1.5, N)

# Create the 3D grid
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# 2. Evaluate Taubin's heart equation across the 3D space
# F = 0 will be the surface of our heart
F = (X**2 + (9/4)*Y**2 + Z**2 - 1)**3 - (X**2)*(Z**3) - (9/80)*(Y**2)*(Z**3)

print("Calculating the isosurface and generating polygons... (please wait a few seconds)")

# 3. Use Marching Cubes to extract vertices and faces (polygons)
verts, faces, normals, values = measure.marching_cubes(F, level=0.0)

# Format the faces array for PyVista (requires indicating the number of points per face at the beginning)
# Since all are triangles, we insert a '3' before each trio of vertices.
faces_pv = np.column_stack((np.full(len(faces), 3), faces)).flatten()

# 4. Build the mesh
mesh = pv.PolyData(verts, faces_pv)
mesh.compute_normals(inplace=True) # Necessary for proper lighting calculation

print(f"Done! The mesh has been generated:")
print(f" -> Polygons (Faces): {mesh.n_cells:,}")
print(f" -> Vertices: {mesh.n_points:,}")

# 5. Rendering
plotter = pv.Plotter()

plotter.add_mesh(
    mesh,
    color='#e30022',       # Intense red
    smooth_shading=True,   # Smooths normals across the millions of polygons
    specular=0.5,          # Shininess
    ambient=0.1            # Base shadow
)

plotter.set_background('#121212') # Dark background to make the model stand out
plotter.add_light(pv.Light(position=(2, 2, 2), focal_point=(0, 0, 0), color='white'))

# Show the window (you can rotate and zoom with the mouse)
plotter.show()