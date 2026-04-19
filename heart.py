"""
Romantic 3D Heart Generator - Professional Edition
--------------------------------------------------
This application generates a high-fidelity 3D model of the Taubin Heart surface,
places a custom geometric text inside, creates a cinematic environment with a 
starfield and pedestal, and animates it with a rhythmic heartbeat and rotation.
"""

import math
import time
import numpy as np
import pyvista as pv
from skimage import measure

# =============================================================================
# --- CONFIGURATION CONSTANTS ---
# =============================================================================
RESOLUTION = 350             # 350 gives ~1.2 million polygons. 
HEART_COLOR = '#FF69B4'      # CAMBIADO A ROSA (Hot Pink) para Anel <3
TEXT_COLOR = '#FFD700'       # Warm Gold
BACKGROUND_COLOR = '#050508' # Deep midnight space blue/black
BASE_ROTATION_SPEED = 0.5    # Degrees per frame
BASE_BEAT_BPM = 60           # Beats per minute

class RomanticHeartScene:
    def __init__(self, resolution: int = RESOLUTION, text: str = "Anel"):
        """
        Initializes the scene components, geometry, and plotter.
        """
        self.resolution = resolution
        self.target_text = text
        
        # Plotter instance for rendering
        self.plotter = pv.Plotter(window_size=[1280, 720])
        self.plotter.title = "A Gift For You"
        
        # State variables for animation and UI
        self.rotation_speed = BASE_ROTATION_SPEED
        self.beat_bpm = BASE_BEAT_BPM
        self.time_elapsed = 0.0
        self.last_time = time.time()
        self.heart_scale = 1.0
        
        # Placeholders for actors
        self.heart_actor = None
        self.text_actor = None
        self.light_key = None

    # =========================================================================
    # --- GEOMETRY GENERATION ---
    # =========================================================================

    def _generate_taubin_heart(self) -> pv.PolyData:
        """
        Evaluates Taubin's algebraic equation and extracts the mesh.
        Equation: (x^2 + 9/4*y^2 + z^2 - 1)^3 - x^2*z^3 - 9/80*y^2*z^3 = 0
        """
        print("[1/5] Calculating Taubin's surface logic... (This may take a moment)")
        x = np.linspace(-1.5, 1.5, self.resolution)
        y = np.linspace(-1.5, 1.5, self.resolution)
        z = np.linspace(-1.5, 1.5, self.resolution)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Evaluate the field
        F = (X**2 + (9/4)*Y**2 + Z**2 - 1)**3 - (X**2)*(Z**3) - (9/80)*(Y**2)*(Z**3)

        print("[2/5] Extracting millions of polygons using Marching Cubes...")
        
        # Tell the algorithm to scale the 350 voxels back down into a 3-unit space
        spacing_val = 3.0 / (self.resolution - 1)
        verts, faces, _, _ = measure.marching_cubes(
            F, 
            level=0.0, 
            spacing=(spacing_val, spacing_val, spacing_val)
        )

        # Format for PyVista (prepend 3 for triangles)
        faces_pv = np.column_stack((np.full(len(faces), 3), faces)).flatten()
        
        heart_mesh = pv.PolyData(verts, faces_pv)
        heart_mesh.compute_normals(inplace=True)
        
        # Center the heart exactly at origin to make scaling/beating easier
        heart_mesh.translate(-np.array(heart_mesh.center), inplace=True)
        
        return heart_mesh

    def _generate_standing_text(self, heart_mesh: pv.PolyData) -> pv.PolyData:
        """
        Creates 3D text, rotates it vertically, and centers it inside the heart.
        """
        print("[3/5] Forging the golden text...")
        text_mesh = pv.Text3D(self.target_text, depth=0.25)

        # Rotate to stand vertically
        text_mesh.rotate_x(90, inplace=True)

        # Center local origin
        text_center = text_mesh.center
        text_mesh.translate([-text_center[0], -text_center[1], -text_center[2]], inplace=True)

        # Scale based on heart proportions
        heart_bounds = heart_mesh.bounds
        heart_height = heart_bounds[5] - heart_bounds[4]
        text_height = text_mesh.bounds[5] - text_mesh.bounds[4]
        
        # We want the text to be roughly 25% of the heart's height
        scale_factor = (heart_height * 0.25) / text_height
        text_mesh.scale(scale_factor, inplace=True)

        # Vertical offset (Taubin's visual center is slightly higher than geometric)
        z_offset = heart_height * 0.12 
        text_mesh.translate([0, 0, z_offset], inplace=True)
        
        return text_mesh

    def _generate_environment(self):
            """
            Creates a cinematic environment: a pedestal and a particle starfield.
            """
            print("[4/5] Building the surrounding environment...")
            
            # 1. The Obsidian Pedestal
            base = pv.Cylinder(center=(0, 0, -1.8), direction=(0, 0, 1), radius=1.0, height=0.2)
            base.compute_normals(inplace=True)
            self.plotter.add_mesh(base, color='#111111', specular=0.5, smooth_shading=True)
            
            # 2. The Golden Ring glowing on the pedestal
            ring = pv.Cylinder(center=(0, 0, -1.68), direction=(0, 0, 1), radius=0.8, height=0.02)
            self.plotter.add_mesh(ring, color=TEXT_COLOR, specular=1.0, ambient=0.5)
            
            # 3. Cinematic Starfield (IMPROVED CARTESIAN VOLUME)
            np.random.seed(42)
            
            # Increased to 4000 since we are filling a much larger volume now
            num_stars = 4000 
            
            # Scatter stars evenly in a massive 30x30x30 cube around the scene
            star_x = np.random.uniform(-15, 15, num_stars)
            star_y = np.random.uniform(-15, 15, num_stars)
            star_z = np.random.uniform(-15, 15, num_stars)
            
            # Calculate the distance of every star from the center (0,0,0)
            distances = np.sqrt(star_x**2 + star_y**2 + star_z**2)
            
            # HOLLOW OUT THE CENTER: 
            # Only keep stars that are further than 6 units away from the heart.
            # This completely removes the dots from the middle of the screen.
            mask = distances > 6.0 
            
            # Apply the mask to filter out the center stars
            star_x = star_x[mask]
            star_y = star_y[mask]
            star_z = star_z[mask]
            
            stars_points = np.column_stack((star_x, star_y, star_z))
            stars_mesh = pv.PolyData(stars_points)
            
            # Render the stars
            self.plotter.add_mesh(
                stars_mesh, 
                color='white', 
                point_size=2.0, 
                render_points_as_spheres=True,
                opacity=0.6
            )

    # =========================================================================
    # --- RENDERING & UI SETUP ---
    # =========================================================================

    def _setup_lighting(self):
        """
        Creates a custom 3-point studio lighting setup.
        """
        # Key Light (Main illumination, slightly warm)
        self.light_key = pv.Light(position=(3, 3, 3), focal_point=(0, 0, 0), color='#FFF5E6')
        self.light_key.intensity = 1.2
        self.plotter.add_light(self.light_key)
                    
        # Fill Light (Cooler light from the opposite side to soften shadows)
        light_fill = pv.Light(position=(-3, -2, 1), focal_point=(0, 0, 0), color='#E6F0FF')
        light_fill.intensity = 0.5
        self.plotter.add_light(light_fill)
        
        # Backlight (Rim lighting to make the ruby pop off the background)
        light_back = pv.Light(position=(0, 5, -2), focal_point=(0, 0, 0), color='white')
        light_back.intensity = 0.8
        self.plotter.add_light(light_back)

    def _setup_ui_widgets(self):
        """
        Adds interactive sliders so the user can control the scene properties.
        """
        # Rotation Speed Slider
        self.plotter.add_slider_widget(
            self._set_rotation_speed,
            rng=[0.0, 2.0],
            value=self.rotation_speed,
            title="Rotation Speed",
            pointa=(0.02, 0.9), pointb=(0.2, 0.9),
            style='modern'
        )
        
        # Heartbeat Speed Slider
        self.plotter.add_slider_widget(
            self._set_beat_speed,
            rng=[0, 150],
            value=self.beat_bpm,
            title="Heartbeat (BPM)",
            pointa=(0.02, 0.78), pointb=(0.2, 0.78),
            style='modern'
        )
        
        # Opacity Slider
        self.plotter.add_slider_widget(
            self._set_opacity,
            rng=[0.1, 1.0],
            value=0.3,
            title="Crystal Opacity",
            pointa=(0.02, 0.66), pointb=(0.2, 0.66),
            style='modern'
        )

    # --- UI Callbacks ---
    def _set_rotation_speed(self, value):
        self.rotation_speed = value

    def _set_beat_speed(self, value):
        self.beat_bpm = value

    def _set_opacity(self, value):
        if self.heart_actor is not None:
            self.heart_actor.GetProperty().SetOpacity(value)

    # =========================================================================
    # --- ANIMATION ENGINE ---
    # =========================================================================

    def _animation_tick(self, step):
        """
        This function is called every few milliseconds by the timer.
        It handles the rotation, the heartbeat scaling, and light pulsing.
        """
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        self.time_elapsed += dt

        # 1. Apply Rotation
        if self.rotation_speed > 0:
            self.plotter.camera.azimuth += self.rotation_speed

        # 2. Apply Heartbeat Physics
        if self.beat_bpm > 0 and self.heart_actor is not None:
            # Convert BPM to frequency (beats per second)
            freq = self.beat_bpm / 60.0
            
            # Create a heartbeat waveform using sine and powers for a "thump" effect
            # sin^6 creates sharp, short peaks rather than a smooth wave
            wave = math.sin(self.time_elapsed * math.pi * freq) ** 6
            
            # Scale goes from 1.0 to 1.06
            target_scale = 1.0 + (0.06 * wave)
            
            # Apply absolute scale to the actors using PyVista's underlying VTK properties
            self.heart_actor.SetScale(target_scale, target_scale, target_scale)
            self.text_actor.SetScale(target_scale, target_scale, target_scale)
            
            # Pulse the light slightly with the beat for a cinematic effect
            if self.light_key is not None:
                self.light_key.intensity = 1.0 + (0.4 * wave)

    # =========================================================================
    # --- MAIN EXECUTION ---
    # =========================================================================

    def build_and_run(self):
        """
        Assembles all components, prepares the renderer, and launches the window.
        """
        # Generate Core Geometries
        heart_mesh = self._generate_taubin_heart()
        text_mesh = self._generate_standing_text(heart_mesh)
        
        print(f"[5/5] Assembling the scene ({heart_mesh.n_cells:,} polygons)...")

        # Configure background and environment
        self.plotter.set_background(BACKGROUND_COLOR)
        self._generate_environment()
        self._setup_lighting()

        # Add the Translucent Ruby Heart
        self.heart_actor = self.plotter.add_mesh(
            heart_mesh,
            color=HEART_COLOR,
            smooth_shading=True,
            specular=1.0,           
            specular_power=50,      # Makes the highlights sharp like glass
            opacity=0.3,
            ambient=0.1
        )

        # Add the Solid Golden Text
        self.text_actor = self.plotter.add_mesh(
            text_mesh,
            color=TEXT_COLOR,
            smooth_shading=True,
            specular=0.8,
            ambient=0.2
        )

        # Add the Romantic Dedication
        self.plotter.add_text(
            "Un abrazo que dure para siempre.\nAnel, te quiero.", 
            position='lower_right', 
            color='#cccccc', 
            font_size=14,
            font='courier'
        )

        # Add UI and Animation hooks
        self._setup_ui_widgets()
        
        # Fire animation loop every 16ms (~60 frames per second)
        self.plotter.add_timer_event(max_steps=200000, duration=16, callback=self._animation_tick)

        print("\n" + "="*50)
        print("Success! Opening the cinematic interaction window.")
        print("Use the sliders on the left to control the magic.")
        print("Rotate: Left Click + Drag | Zoom: Scroll Wheel")
        print("="*50)
        
        # Initialize the last_time right before showing to prevent a massive first jump
        self.last_time = time.time()
        
        # Launch the PyVista window
        self.plotter.show()


# =============================================================================
# --- ENTRY POINT ---
# =============================================================================
if __name__ == "__main__":
    app = RomanticHeartScene(resolution=350, text="Anel")
    app.build_and_run()