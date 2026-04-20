"""
Felix (SKZ) 3D Heart Generator - BbokAri Edition
Version: 2.0 (Spacebar Pause Update)

This script generates a 3D Taubin heart surface with a golden text overlay,
mimicking the aesthetic of Stray Kids' BbokAri. It features dynamic lighting,
a heartbeat pulse, and interactive rotation.

Key Update: Press 'Space' to toggle the orbital rotation while keeping the heartbeat active.
"""

import math
import time
import sys # <-- ADDED: For handling the command line test flag
import unittest # <-- ADDED: Python's built-in testing framework
import numpy as np
import pyvista as pv
from skimage import measure

# --- Global Configuration ---
RESOLUTION = 350             # Detail of the 3D mesh (Higher = smoother, but slower)
HEART_COLOR = '#EABECD'      # BbokAri Pink
TEXT_COLOR = '#FFD700'       # Gold
BACKGROUND_COLOR = '#050508' # Deep Space
BASE_ROTATION_SPEED = 0.5    # Default degrees per frame
BASE_BEAT_BPM = 60           # Default heartbeat frequency


class StrayKidsHeartScene:
    def __init__(self, resolution=RESOLUTION, text="Anel", texture_path=None):
        """
        Initializes the scene components, engine states, and plotter.
        """
        self.resolution = resolution
        self.target_text = text
        self.texture_path = texture_path

        # Plotter Setup
        self.plotter = pv.Plotter(window_size=[1280, 720])
        self.plotter.title = "A 5-Star Michelin Gift - BbokAri Loves You"

        # Animation State Variables
        self.rotation_speed = BASE_ROTATION_SPEED
        self.beat_bpm = BASE_BEAT_BPM
        self.is_rotating = True  # Flag to control spacebar rotation toggle
        self.time_elapsed = 0.0
        self.last_time = time.time()

        # Actor references (objects in the 3D scene)
        self.heart_actor = None
        self.heart_base_actor = None
        self.text_actor = None
        self.light_key = None

    def _generate_taubin_heart(self):
        """
        Uses the Taubin algebraic heart equation: 
        (x^2 + (9/4)y^2 + z^2 - 1)^3 - x^2z^3 - (9/80)y^2z^3 = 0
        """
        print("[1/5] Calculating Taubin surface...")
        x = np.linspace(-1.5, 1.5, self.resolution)
        y = np.linspace(-1.5, 1.5, self.resolution)
        z = np.linspace(-1.5, 1.5, self.resolution)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # The mathematical definition of the heart shape
        F = (X**2 + (9/4)*Y**2 + Z**2 - 1)**3 - (X**2)*(Z**3) - (9/80)*(Y**2)*(Z**3)

        print("[2/5] Extracting polygons (marching cubes)...")
        spacing_val = 3.0 / (self.resolution - 1)
        # Convert volume data to a mesh of triangles
        verts, faces, _, _ = measure.marching_cubes(F, level=0.0, spacing=(spacing_val, spacing_val, spacing_val))

        # Reformat faces for PyVista (prepend number of points per face, which is 3)
        faces_pv = np.column_stack((np.full(len(faces), 3), faces)).flatten()
        heart_mesh = pv.PolyData(verts, faces_pv)
        heart_mesh.compute_normals(inplace=True)
        heart_mesh.translate(-np.array(heart_mesh.center), inplace=True)

        # --- UV Mapping for Texture placement ---
        pts = heart_mesh.points
        x_pts, z_pts = pts[:, 0], pts[:, 2]
        x_min, x_max = np.min(x_pts), np.max(x_pts)
        z_min, z_max = np.min(z_pts), np.max(z_pts)

        # Normalize UVs and apply a zoom factor to center the image on the heart face
        u_base = 1.0 - ((x_pts - x_min) / (x_max - x_min))
        v_base = (z_pts - z_min) / (z_max - z_min)
        
        zoom_factor = 2.5
        u = ((u_base - 0.5) / zoom_factor) + 0.5
        v = ((v_base - 0.5) / zoom_factor) + 0.5
        
        heart_mesh.active_texture_coordinates = np.column_stack((np.clip(u, 0, 1), np.clip(v, 0, 1)))
        return heart_mesh

    def _generate_standing_text(self, heart_mesh):
        """Creates a 3D text mesh positioned on the surface of the heart."""
        print("[3/5] Forging golden text...")
        text_mesh = pv.Text3D(self.target_text, depth=0.25)
        text_mesh.rotate_x(90, inplace=True)
        
        # Center the text
        tc = text_mesh.center
        text_mesh.translate([-tc[0], -tc[1], -tc[2]], inplace=True)
        
        # Scale relative to heart size
        heart_height = heart_mesh.bounds[5] - heart_mesh.bounds[4]
        text_height = text_mesh.bounds[5] - text_mesh.bounds[4]
        scale_factor = (heart_height * 0.25) / text_height
        text_mesh.scale(scale_factor, inplace=True)
        
        # Offset text slightly forward so it sits on the surface
        text_mesh.translate([0, 0, heart_height * 0.12], inplace=True)
        return text_mesh

    def _generate_environment(self):
        """Creates the display pedestal and the starfield background."""
        print("[4/5] Building SKZOO environment...")
        
        # Pedestal
        base = pv.Cylinder(center=(0, 0, -1.8), direction=(0, 0, 1), radius=1.0, height=0.2)
        self.plotter.add_mesh(base, color='#111111', specular=0.5, smooth_shading=True)
        
        # Golden Ring on pedestal
        ring = pv.Cylinder(center=(0, 0, -1.68), direction=(0, 0, 1), radius=0.8, height=0.02)
        self.plotter.add_mesh(ring, color=TEXT_COLOR, specular=1.0, ambient=0.5)
        
        # Procedural Starfield
        np.random.seed(42)
        num_stars = 4000
        stars_pos = np.random.uniform(-15, 15, (num_stars, 3))
        dist = np.linalg.norm(stars_pos, axis=1)
        mask = dist > 6.0 # Don't place stars inside the heart area
        stars_mesh = pv.PolyData(stars_pos[mask])
        self.plotter.add_mesh(stars_mesh, color='white', point_size=2.0, render_points_as_spheres=True, opacity=0.6)

    def _setup_lighting(self):
        """Configures a 3-point lighting setup for cinematic depth."""
        self.light_key = pv.Light(position=(3, 3, 3), color='#FFF5E6', intensity=1.2)
        self.plotter.add_light(self.light_key)
        
        light_fill = pv.Light(position=(-3, -2, 1), color='#E6F0FF', intensity=0.5)
        self.plotter.add_light(light_fill)
        
        light_back = pv.Light(position=(0, 5, -2), color='white', intensity=0.8)
        self.plotter.add_light(light_back)

    def _setup_ui_widgets(self):
        """Adds interactive sliders to the screen."""
        self.plotter.add_slider_widget(
            self._set_rotation_speed, rng=[0.0, 2.0], value=self.rotation_speed,
            title="Rotation Speed", pointa=(0.02, 0.90), pointb=(0.2, 0.90), style='modern')
        
        self.plotter.add_slider_widget(
            self._set_beat_speed, rng=[0, 150], value=self.beat_bpm,
            title="Heartbeat (BPM)", pointa=(0.02, 0.78), pointb=(0.2, 0.78), style='modern')
        
        self.plotter.add_slider_widget(
            self._set_opacity, rng=[0.1, 1.0], value=1.0,
            title="Opacity", pointa=(0.02, 0.66), pointb=(0.2, 0.66), style='modern')

    # Slider Callbacks
    def _set_rotation_speed(self, value): self.rotation_speed = value
    def _set_beat_speed(self, value):     self.beat_bpm = value
    def _set_opacity(self, value):
        if self.heart_actor: self.heart_actor.GetProperty().SetOpacity(value)

    def _toggle_rotation(self):
        """Toggle function triggered by Spacebar."""
        self.is_rotating = not self.is_rotating
        state = "RESUMED" if self.is_rotating else "PAUSED"
        print(f"[Input] Rotation {state}")

    def _animation_tick(self, step):
        """The main loop handling movement and pulse logic."""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        self.time_elapsed += dt

        # Handle Orbital Rotation (Camera Azimuth)
        # Only rotates if global flag is true AND slider is > 0
        if self.is_rotating and self.rotation_speed > 0:
            self.plotter.camera.azimuth += self.rotation_speed

        # Handle Heartbeat Pulse (Independent of rotation)
        if self.beat_bpm > 0 and self.heart_actor is not None:
            freq = self.beat_bpm / 60.0
            # Use a high power of sine to create a sharp "thump-thump" effect
            wave = math.sin(self.time_elapsed * math.pi * freq) ** 6
            target_scale = 1.0 + (0.06 * wave)
            
            # Apply scaling to mesh actors
            self.heart_actor.SetScale(target_scale, target_scale, target_scale)
            if self.heart_base_actor:
                self.heart_base_actor.SetScale(target_scale, target_scale, target_scale)
            self.text_actor.SetScale(target_scale, target_scale, target_scale)
            
            # Pulse lighting intensity with the beat
            if self.light_key:
                self.light_key.intensity = 1.0 + (0.4 * wave)

    def build_and_run(self):
        """Assembles all components and starts the PyVista event loop."""
        heart_mesh = self._generate_taubin_heart()
        text_mesh  = self._generate_standing_text(heart_mesh)

        print(f"[5/5] Assembling scene ({heart_mesh.n_cells:,} polygons)...")

        self.plotter.set_background(BACKGROUND_COLOR)
        self._generate_environment()
        self._setup_lighting()

        # Load Texture
        heart_texture = None
        if self.texture_path:
            try:
                heart_texture = pv.read_texture(self.texture_path)
                heart_texture.SetWrap(0) # Clamp to border
                print(f"[OK] Texture loaded: {self.texture_path}")
            except Exception as e:
                print(f"[!] Texture load failed: {e}")

        # Main Heart Actor
        self.heart_actor = self.plotter.add_mesh(
            heart_mesh,
            texture=heart_texture,
            color=HEART_COLOR,
            smooth_shading=True,
            specular=0.4,
            specular_power=25,
            opacity=1.0,
            ambient=0.9, 
            diffuse=0.6,
        )

        # Text Actor
        self.text_actor = self.plotter.add_mesh(
            text_mesh, color=TEXT_COLOR, smooth_shading=True, specular=0.8, ambient=0.2)

        # Footer Text
        self.plotter.add_text(
            "Anel 🐥 | [Space] to Pause Rotation",
            position='lower_right', color='#cccccc', font_size=12, font='courier')

        # Register UI and Key Events
        self._setup_ui_widgets()
        self.plotter.add_key_event("space", self._toggle_rotation)
        
        # Start Animation Timer
        self.plotter.add_timer_event(max_steps=200000, duration=16, callback=self._animation_tick)

        print("\n" + "="*50)
        print("BbokAri is ready!")
        print("- Press SPACE to stop/start rotation.")
        print("- Use Sliders to adjust the pulse and speed.")
        print("="*50)

        self.last_time = time.time()
        self.plotter.show()


# --- NEW: Testing Suite ---
class TestStrayKidsHeartScene(unittest.TestCase):
    def setUp(self):
        # We initialize with a very low resolution (e.g., 20) just for testing.
        # This makes the math calculate instantly without freezing the test runner.
        self.app = StrayKidsHeartScene(resolution=20, text="Test")

    def test_initial_state(self):
        """Ensures the app starts with the correct default states."""
        self.assertTrue(self.app.is_rotating)
        self.assertEqual(self.app.rotation_speed, BASE_ROTATION_SPEED)
        self.assertEqual(self.app.beat_bpm, BASE_BEAT_BPM)

    def test_toggle_rotation(self):
        """Simulates pressing the spacebar to ensure the logic works."""
        self.app._toggle_rotation()
        self.assertFalse(self.app.is_rotating, "Rotation should be False after one toggle.")
        
        self.app._toggle_rotation()
        self.assertTrue(self.app.is_rotating, "Rotation should be True after second toggle.")

    def test_mesh_generation(self):
        """Checks if the Taubin equation successfully generates 3D geometry."""
        mesh = self.app._generate_taubin_heart()
        
        # Verify it returns a PyVista PolyData object
        self.assertIsInstance(mesh, pv.PolyData)
        
        # Verify it actually contains 3D points
        self.assertGreater(mesh.n_points, 0, "Mesh failed to generate points.")
        
        # Verify UV coordinates were successfully applied
        self.assertIsNotNone(mesh.active_texture_coordinates, "UV mapping failed.")

    def tearDown(self):
        # Clean up the plotter from memory so tests don't leak resources
        self.app.plotter.close()


if __name__ == "__main__":
    # --- NEW: Command Line Logic ---
    # This allows you to run the file normally, OR run it in "test mode"
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running tests for BbokAri Heart Generator...")
        sys.argv.pop() # Remove the flag so unittest doesn't get confused
        unittest.main()
    else:
        # Note: Ensure "02texture.jpg" exists in your working directory for the texture to apply.
        app = StrayKidsHeartScene(resolution=350, text="Anel", texture_path="02texture.jpg")
        app.build_and_run()