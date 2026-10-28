"""Bind RP1's plane MDL on Isaac Lab versions that only copy diffuse color."""

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter


class ReferencePlaneImporter(TerrainImporter):
    def import_ground_plane(self, name, size=(2.0e6, 2.0e6)):
        material = self.cfg.visual_material
        # The installed importer does not bind a plane's visual_material itself.
        self.cfg.visual_material = None
        try:
            super().import_ground_plane(name, size)
        finally:
            self.cfg.visual_material = material
        if material is not None:
            terrain_path = f"{self.cfg.prim_path}/{name}"
            material_path = f"{terrain_path}/rp1_visual_material"
            material.func(material_path, material)
            sim_utils.bind_visual_material(terrain_path, material_path)
