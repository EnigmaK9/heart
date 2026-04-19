"""
Felix (SKZ) 3D Heart Generator - BbokAri Edition
"""

import math
import time
import numpy as np
import pyvista as pv
from skimage import measure

RESOLUTION = 350
HEART_COLOR = '#EABECD'  # Rosa exacto de la referencia del usuario
TEXT_COLOR = '#FFD700'
BACKGROUND_COLOR = '#050508'
BASE_ROTATION_SPEED = 0.5
BASE_BEAT_BPM = 60


class StrayKidsHeartScene:
    def __init__(self, resolution=RESOLUTION, text="Anel", texture_path=None):
        self.resolution = resolution
        self.target_text = text
        self.texture_path = texture_path

        self.plotter = pv.Plotter(window_size=[1280, 720])
        self.plotter.title = "A 5-Star Michelin Gift - BbokAri Loves You"

        self.rotation_speed = BASE_ROTATION_SPEED
        self.beat_bpm = BASE_BEAT_BPM
        self.time_elapsed = 0.0
        self.last_time = time.time()

        self.heart_actor = None
        self.heart_base_actor = None
        self.text_actor = None
        self.light_key = None

    def _generate_taubin_heart(self):
        print("[1/5] Calculando superficie Taubin...")
        x = np.linspace(-1.5, 1.5, self.resolution)
        y = np.linspace(-1.5, 1.5, self.resolution)
        z = np.linspace(-1.5, 1.5, self.resolution)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        F = (X**2 + (9/4)*Y**2 + Z**2 - 1)**3 - (X**2)*(Z**3) - (9/80)*(Y**2)*(Z**3)

        print("[2/5] Extrayendo polígonos (marching cubes)...")
        spacing_val = 3.0 / (self.resolution - 1)
        verts, faces, _, _ = measure.marching_cubes(F, level=0.0, spacing=(spacing_val, spacing_val, spacing_val))

        faces_pv = np.column_stack((np.full(len(faces), 3), faces)).flatten()
        heart_mesh = pv.PolyData(verts, faces_pv)
        heart_mesh.compute_normals(inplace=True)
        heart_mesh.translate(-np.array(heart_mesh.center), inplace=True)

        # UV PLANAR FRONTAL con margen y clamp
        pts = heart_mesh.points
        x_pts = pts[:, 0]
        z_pts = pts[:, 2]

        x_min, x_max = np.min(x_pts), np.max(x_pts)
        z_min, z_max = np.min(z_pts), np.max(z_pts)

        margin = -0.12
        x_range = (x_max - x_min) * (1.0 + 2 * abs(margin))
        z_range = (z_max - z_min) * (1.0 + 2 * abs(margin))

        u = (x_pts - (x_min - abs(margin) * (x_max - x_min))) / x_range
        u = 1.0 - u  # flip horizontal
        v = (z_pts - (z_min - abs(margin) * (z_max - z_min))) / z_range

        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)

        heart_mesh.active_texture_coordinates = np.column_stack((u, v))
        return heart_mesh

    def _generate_standing_text(self, heart_mesh):
        print("[3/5] Forjando el texto dorado...")
        text_mesh = pv.Text3D(self.target_text, depth=0.25)
        text_mesh.rotate_x(90, inplace=True)
        tc = text_mesh.center
        text_mesh.translate([-tc[0], -tc[1], -tc[2]], inplace=True)
        heart_height = heart_mesh.bounds[5] - heart_mesh.bounds[4]
        text_height = text_mesh.bounds[5] - text_mesh.bounds[4]
        scale_factor = (heart_height * 0.25) / text_height
        text_mesh.scale(scale_factor, inplace=True)
        text_mesh.translate([0, 0, heart_height * 0.12], inplace=True)
        return text_mesh

    def _generate_environment(self):
        print("[4/5] Construyendo el entorno SKZOO...")
        base = pv.Cylinder(center=(0, 0, -1.8), direction=(0, 0, 1), radius=1.0, height=0.2)
        base.compute_normals(inplace=True)
        self.plotter.add_mesh(base, color='#111111', specular=0.5, smooth_shading=True)
        ring = pv.Cylinder(center=(0, 0, -1.68), direction=(0, 0, 1), radius=0.8, height=0.02)
        self.plotter.add_mesh(ring, color=TEXT_COLOR, specular=1.0, ambient=0.5)
        np.random.seed(42)
        num_stars = 4000
        sx = np.random.uniform(-15, 15, num_stars)
        sy = np.random.uniform(-15, 15, num_stars)
        sz = np.random.uniform(-15, 15, num_stars)
        dist = np.sqrt(sx**2 + sy**2 + sz**2)
        mask = dist > 6.0
        stars_mesh = pv.PolyData(np.column_stack((sx[mask], sy[mask], sz[mask])))
        self.plotter.add_mesh(stars_mesh, color='white', point_size=2.0,
                              render_points_as_spheres=True, opacity=0.6)

    def _setup_lighting(self):
        self.light_key = pv.Light(position=(3, 3, 3), focal_point=(0, 0, 0), color='#FFF5E6')
        self.light_key.intensity = 1.2
        self.plotter.add_light(self.light_key)
        light_fill = pv.Light(position=(-3, -2, 1), focal_point=(0, 0, 0), color='#E6F0FF')
        light_fill.intensity = 0.5
        self.plotter.add_light(light_fill)
        light_back = pv.Light(position=(0, 5, -2), focal_point=(0, 0, 0), color='white')
        light_back.intensity = 0.8
        self.plotter.add_light(light_back)

    def _setup_ui_widgets(self):
        self.plotter.add_slider_widget(
            self._set_rotation_speed, rng=[0.0, 2.0], value=self.rotation_speed,
            title="Rotation Speed", pointa=(0.02, 0.90), pointb=(0.2, 0.90), style='modern')
        self.plotter.add_slider_widget(
            self._set_beat_speed, rng=[0, 150], value=self.beat_bpm,
            title="Heartbeat (BPM)", pointa=(0.02, 0.78), pointb=(0.2, 0.78), style='modern')
        self.plotter.add_slider_widget(
            self._set_opacity, rng=[0.1, 1.0], value=1.0,
            title="Opacity", pointa=(0.02, 0.66), pointb=(0.2, 0.66), style='modern')

    def _set_rotation_speed(self, value): self.rotation_speed = value
    def _set_beat_speed(self, value):     self.beat_bpm = value
    def _set_opacity(self, value):
        if self.heart_actor: self.heart_actor.GetProperty().SetOpacity(value)

    def _animation_tick(self, step):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        self.time_elapsed += dt

        if self.rotation_speed > 0:
            self.plotter.camera.azimuth += self.rotation_speed

        if self.beat_bpm > 0 and self.heart_actor is not None:
            freq = self.beat_bpm / 60.0
            wave = math.sin(self.time_elapsed * math.pi * freq) ** 6
            target_scale = 1.0 + (0.06 * wave)
            self.heart_actor.SetScale(target_scale, target_scale, target_scale)
            if self.heart_base_actor:
                self.heart_base_actor.SetScale(target_scale, target_scale, target_scale)
            self.text_actor.SetScale(target_scale, target_scale, target_scale)
            if self.light_key:
                self.light_key.intensity = 1.0 + (0.4 * wave)

    def build_and_run(self):
        heart_mesh = self._generate_taubin_heart()
        text_mesh  = self._generate_standing_text(heart_mesh)

        print(f"[5/5] Ensamblando la escena ({heart_mesh.n_cells:,} polígonos)...")

        self.plotter.set_background(BACKGROUND_COLOR)
        self._generate_environment()
        self._setup_lighting()

        heart_texture = None
        if self.texture_path:
            try:
                heart_texture = pv.read_texture(self.texture_path)
                heart_texture.SetWrap(0)  # EDGE_CLAMP: borde rosa, nunca negro
                print(f"[OK] Textura cargada: {self.texture_path}")
            except Exception as e:
                print(f"[!] No se pudo cargar la textura: {e}")

        if heart_texture:
            # UNA SOLA MALLA: textura con ambient=1.0 para que NUNCA haya zonas oscuras.
            # ambient=1.0 significa que el color base (HEART_COLOR rosa) ilumina uniformemente
            # todo lo que la textura no cubre, eliminando el checkerboard gris de los bordes.
            self.heart_base_actor = None
            self.heart_actor = self.plotter.add_mesh(
                heart_mesh,
                texture=heart_texture,
                color=HEART_COLOR,
                smooth_shading=True,
                specular=0.4,
                specular_power=25,
                opacity=1.0,
                ambient=1.0,   # <-- CLAVE: iluminación uniforme, cero negro
                diffuse=0.6,
            )
        else:
            self.heart_base_actor = None
            self.heart_actor = self.plotter.add_mesh(
                heart_mesh,
                color=HEART_COLOR,
                smooth_shading=True,
                specular=1.0,
                specular_power=60,
                opacity=1.0,
                ambient=1.0,
            )

        self.text_actor = self.plotter.add_mesh(
            text_mesh, color=TEXT_COLOR, smooth_shading=True, specular=0.8, ambient=0.2)

        self.plotter.add_text(
            "Anel 🐥 ",
            position='lower_right', color='#cccccc', font_size=14, font='courier')

        self._setup_ui_widgets()
        self.plotter.add_timer_event(max_steps=200000, duration=16, callback=self._animation_tick)

        print("\n" + "="*50)
        print("¡Listo! Ventana de interacción abierta.")
        print("Rotar: Click izq + Arrastrar | Zoom: Rueda")
        print("="*50)

        self.last_time = time.time()
        self.plotter.show()


if __name__ == "__main__":
    app = StrayKidsHeartScene(resolution=350, text="Anel", texture_path="dgHXD.jpg")
    app.build_and_run()
