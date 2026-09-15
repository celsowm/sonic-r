import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sonic_exporter", ROOT / "tools" / "export_sonicr_character.py")
assert SPEC and SPEC.loader
EXPORTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPORTER
SPEC.loader.exec_module(EXPORTER)


def make_model() -> bytes:
    data = bytearray(struct.pack("<i", 3))
    for vertex in ((0, 0, 0), (16, 0, 0), (0, 16, 0)):
        data.extend(struct.pack("<hhhhhhBBBB", *vertex, 0, 0, 16384, 255, 255, 255, 0))
    data.extend(struct.pack("<i", 1))
    data.extend(struct.pack("<HHHBBBBBBBBh", 0, 1, 2, 0, 0, 255, 0, 0, 255, 0, 0, 0))
    data.extend(struct.pack("<i", 0))
    return bytes(data)


class SonicRExporterTests(unittest.TestCase):

    def test_default_animation_tick_rate_matches_game_simulation(self):
        self.assertEqual(EXPORTER.FPS_DEFAULT, 30.0)

    def test_barycentric_weights_reconstruct_point(self):
        triangle = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        weights = EXPORTER.barycentric((0.5, 0.25), triangle)
        self.assertEqual(tuple(round(value, 6) for value in weights), (0.5, 0.25, 0.25))
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_frame_stream_drops_triggers_and_jump_operands(self) -> None:
        self.assertEqual(EXPORTER.frame_references([1, 4100, -2, 2, 5, -1, 2]), [1, 4, 5])

    def test_raw_rgb_is_a_png(self) -> None:
        image = EXPORTER.png_rgb(2, 1, bytes((255, 0, 0, 0, 255, 0)))
        self.assertEqual(image[:8], b"\x89PNG\r\n\x1a\n")

    def test_raw_color_key_is_transparent_rgba(self) -> None:
        raw = bytearray(256 * 256 * 3)
        raw[0:3] = bytes((0, 255, 0))
        raw[3:6] = bytes((7, 248, 7))
        raw[6:9] = bytes((8, 248, 8))
        rgba = EXPORTER.raw_rgba(bytes(raw))
        self.assertEqual(rgba[3], 0)
        self.assertEqual(rgba[7], 0)
        self.assertEqual(rgba[11], 255)
        image = EXPORTER.png_rgba(1, 1, bytes((1, 2, 3, 0)))
        self.assertEqual(image[25], 6)  # PNG color type RGBA

    def test_renderer_model_patches_remap_pages_and_amy_vehicle(self) -> None:
        vertices = [EXPORTER.Vertex((0, 0, 0), (0, 0, 1), (255, 255, 255, 255))] * 3
        polygons = [EXPORTER.Polygon(0, [0, 1, 2], [(0, 0)] * 3, 0, i) for i in range(117)]
        limbs = [EXPORTER.Limb(vertices, polygons)]
        patched = EXPORTER.apply_renderer_model_patches(EXPORTER.CHARACTER_BY_SLUG["amy"], limbs)
        self.assertEqual(patched[0].atlas, 0)
        self.assertEqual(patched[1].atlas, 1)
        self.assertEqual(patched[34].atlas, 1)
        self.assertTrue(patched[34].double_sided)
        self.assertEqual(patched[34].uvs[0], (111.5 / 256.0, 30.5 / 256.0))

    def test_tails_runtime_geometry_has_24_double_sided_quads(self) -> None:
        vertex = EXPORTER.Vertex((0, 0, 0), (0, 0, 1), (255, 255, 255, 255))
        limbs = [EXPORTER.Limb([vertex] * 340, [])]
        patched = EXPORTER.apply_renderer_model_patches(EXPORTER.CHARACTER_BY_SLUG["tails"], limbs)
        self.assertEqual(len(patched), 24)
        self.assertTrue(all(p.double_sided and p.corner_refs for p in patched))
        self.assertEqual(len({ref for p in patched for ref in p.corner_refs or []}), 48)

    def test_baked_tiles_keep_a_gutter_inside_the_cell(self) -> None:
        vertex = EXPORTER.Vertex((0, 0, 0), (0, 0, 1), (255, 255, 255, 255))
        limb = EXPORTER.Limb([vertex] * 3, [])
        polygon = EXPORTER.Polygon(0, [0, 1, 2], [(0.1, 0.1)] * 3, 0, 0)
        raw = bytes(256 * 256 * 3)
        png, uv = EXPORTER.bake_add_signed_atlas([limb], [polygon], {}, set(), [raw, raw], [[(128, 128, 128)] * 3], tile_size=8, gutter=2)
        self.assertEqual(png[25], 6)
        self.assertEqual(uv[0][0][0], 2.5 / 8.0)

    def test_identity_bone_becomes_a_unit_quaternion(self) -> None:
        matrix = EXPORTER.mat_mul(EXPORTER.transpose(EXPORTER.bone_matrix((0, 0, 0))), EXPORTER.BASE_BONE_INVERSE)
        quaternion = EXPORTER.quat_from_matrix(matrix)
        self.assertAlmostEqual(sum(value * value for value in quaternion), 1.0)

    def test_fixed_point_normal_becomes_a_unit_vector(self) -> None:
        normal = EXPORTER.unit_vector((0.0, 0.0, 4096.0))
        self.assertEqual(normal, (0.0, 0.0, 1.0))
        normal = EXPORTER.unit_vector((2048.0, 2048.0, 2048.0))
        self.assertAlmostEqual(sum(value * value for value in normal), 1.0)

    def test_generated_glb_has_embedded_images_skin_and_animation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            general = root / "GENERAL"
            objects = root / "BIN" / "OBJECTS" / "SONIC"
            general.mkdir(parents=True)
            objects.mkdir(parents=True)
            pixels = bytes(256 * 256 * 3)
            (general / "PLAYER00.RAW").write_bytes(pixels)
            (general / "PLAYER01.RAW").write_bytes(pixels)
            (objects / "SONIC_H.BIN").write_bytes(make_model())
            frames = struct.pack("<i", 1) + b"".join(struct.pack("<iiiiii", 0, 0, 0, 0, 0, 0) for _ in range(120))
            (objects / "SAN1.BIN").write_bytes(frames)
            for index in range(2, 15):
                (objects / f"SAN{index}.BIN").write_bytes(struct.pack("<i", 1))
            output = root / "sonic.glb"
            report = EXPORTER.build_glb(EXPORTER.CHARACTER_BY_SLUG["sonic"], root, output, 60.0, 1 / 16)
            self.assertEqual(report["images"], 2)
            self.assertEqual(report["joints"], 1)
            self.assertEqual(report["animations"], 17)
            glb = output.read_bytes()
            self.assertTrue(glb.startswith(b"glTF"))
            json_length = struct.unpack_from("<I", glb, 12)[0]
            document = json.loads(glb[20:20 + json_length])
            self.assertTrue(all("COLOR_0" not in primitive["attributes"]
                                for primitive in document["meshes"][0]["primitives"]))


if __name__ == "__main__":
    unittest.main()
