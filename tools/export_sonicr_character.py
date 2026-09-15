#!/usr/bin/env python3
"""Export Sonic R PC character assets to self-contained GLB files.

The game stores every character as independently transformed rigid limbs rather
than as a parented skeletal hierarchy.  This tool exposes those limbs as a glTF
skin whose joints share one root, which retains the original animation exactly
without inventing a rig.

Only game data supplied by the user is read.  Nothing extracted from an ISO is
kept: ISO extraction is staged in a temporary directory and the only persistent
result is the requested GLB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCALE = 1.0 / 16.0
# The PC game advances a character stream once per 30 Hz simulation tick
# (race_setup.c calls WaitForFrameCap after each TickPlayerAnimation).  Sampling
# those entries at 60 Hz makes every clip play at double speed in glTF viewers.
FPS_DEFAULT = 30.0

ANIMATION_NAMES = {
    0: "idle", 1: "jump", 2: "still", 3: "turn_left", 4: "turn_right",
    # animation.c gives these streams their gameplay meanings: 6 is the win
    # pose and 7 is the slumped lose pose.  The older anim_data.c comments
    # call them "prerace" and "celebrate", which is misleading in an export.
    5: "brake", 6: "victory", 7: "defeat", 8: "trick_left",
    9: "trick_right", 10: "landing", 11: "special", 12: "boost",
    13: "char_specific", 14: "falling", 15: "highspeed", 16: "unused",
    17: "intro",
}


@dataclass(frozen=True)
class CharacterSpec:
    slug: str
    display_name: str
    object_dir: str
    model_file: str
    animation_prefix: str
    animation_count: int
    animation_table: str
    char_id: int


CHARACTERS = (
    CharacterSpec("sonic", "Sonic", "SONIC", "SONIC_H.BIN", "SAN", 14, "Sonic", 0),
    CharacterSpec("tails", "Tails", "TAILS", "TAILS_H.BIN", "TAN", 15, "Tails", 1),
    CharacterSpec("knuckles", "Knuckles", "KNUCKLES", "KNUCK_H.BIN", "KAN", 16, "Knuckles", 2),
    CharacterSpec("amy", "Amy", "AMY", "AMY_H3.BIN", "AAN", 11, "Amy", 3),
    CharacterSpec("eggman", "Eggman", "ROBOTNIK", "ROBOTZ.BIN", "RAN", 9, "Eggman", 4),
    CharacterSpec("metal-sonic", "Metal Sonic", "MSONIC", "MSONICZ.BIN", "MSAN", 13, "Metal", 5),
    CharacterSpec("tails-doll", "Tails Doll", "DTAILS", "DTAILSZ.BIN", "DTAN", 10, "TailsDoll", 6),
    CharacterSpec("metal-knuckles", "Metal Knuckles", "MKNUCK", "MKNUCKZ.BIN", "MKAN", 13, "MKnux", 7),
    CharacterSpec("egg-robo", "Egg Robo", "MROBOT", "MROBOTZ.BIN", "MRAN", 14, "EggRobo", 8),
    CharacterSpec("super-sonic", "Super Sonic", "SSONIC", "SSONICZ.BIN", "SSAN", 14, "Super", 9),
)
CHARACTER_BY_SLUG = {item.slug: item for item in CHARACTERS}

# Sonic R's InstallShield 9 cabinet keeps the payload bytes intact but encrypts
# its file names.  These are *cabinet record numbers*, verified against the
# original executable's directory layout and model signatures; the actual file
# is located by its MD5 stored in the cabinet header.  This lets --iso build a
# normal named data tree without executing the game's installer.
INSTALLER_CHARACTER_RECORDS = {
    "amy": (82, 83, range(71, 82)),
    "tails-doll": (95, 94, range(96, 106)),
    "knuckles": (139, 140, range(123, 139)),
    "metal-knuckles": (155, 154, range(141, 154)),
    "egg-robo": (171, 170, range(156, 170)),
    "metal-sonic": (186, 185, range(172, 185)),
    "eggman": (241, 240, range(231, 240)),
    "sonic": (256, 257, range(242, 256)),
    "super-sonic": (273, 272, range(258, 272)),
    "tails": (274, 275, range(276, 291)),
}
INSTALLER_PLAYER_TEXTURE_RECORDS = {"PLAYER00.RAW": 432, "PLAYER01.RAW": 434}


@dataclass
class Vertex:
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    color: tuple[int, int, int, int]


@dataclass
class Polygon:
    limb: int
    indices: list[int]
    uvs: list[tuple[float, float]]
    atlas: int
    ordinal: int
    # Original model flags (before the loader's flags*2 conversion). Bit 1
    # marks a genuinely double-sided primitive; bit 0 is the quad winding
    # exception for which the game submits both halves when either faces the
    # camera.
    flags: int = 0
    # Runtime-generated polygons (the Tails tail) can join vertices from
    # different limbs. Normal model polygons leave this as None.
    corner_refs: list[tuple[int, int]] | None = None

    @property
    def double_sided(self) -> bool:
        return bool(self.flags & 2) or (len(self.indices) == 4 and bool(self.flags & 1))

    def refs(self) -> list[tuple[int, int]]:
        if self.corner_refs is not None:
            return self.corner_refs
        return [(self.limb, index) for index in self.indices]


@dataclass
class Limb:
    vertices: list[Vertex]
    polygons: list[Polygon]


def die(message: str) -> None:
    raise RuntimeError(message)


def case_path(root: Path, *parts: str) -> Path:
    """Resolve a relative path case-insensitively, including ISO9660 installs."""
    current = root
    for part in parts:
        if not current.is_dir():
            die(f"Diretório ausente ao procurar dados: {current}")
        wanted = part.casefold()
        matches = [child for child in current.iterdir() if child.name.casefold() == wanted]
        if not matches:
            die(f"Arquivo de dados ausente: {'/'.join(parts)} em {root}")
        current = matches[0]
    return current


def looks_like_data_root(path: Path) -> bool:
    try:
        return (case_path(path, "GENERAL", "PLAYER00.RAW").is_file()
                and case_path(path, "GENERAL", "PLAYER01.RAW").is_file()
                and case_path(path, "BIN", "OBJECTS").is_dir())
    except RuntimeError:
        return False


def find_data_root(base: Path) -> Path:
    if looks_like_data_root(base):
        return base
    for directory, _, files in __import__("os").walk(base):
        if any(name.casefold() == "player00.raw" for name in files):
            candidate = Path(directory).parent
            if looks_like_data_root(candidate):
                return candidate
    die("Não encontrei GENERAL/PLAYER00.RAW e BIN/OBJECTS nos dados extraídos.")


def run_7z(args: list[str], description: str) -> None:
    executable = shutil.which("7z") or shutil.which("7zz")
    if not executable:
        die("7-Zip não foi encontrado no PATH; use --data-dir ou instale o 7-Zip.")
    completed = subprocess.run([executable, *args], text=True, capture_output=True)
    if completed.returncode != 0:
        die(f"Falha ao {description} com 7-Zip:\n{completed.stdout}\n{completed.stderr}")


def find_unpacker(explicit: Path | None = None) -> str | None:
    """Find the optional InstallShield extractor without assuming a C: path."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    env_value = __import__("os").environ.get("SONICR_UNPACKER")
    if env_value:
        candidates.append(Path(env_value))
    for name in ("Unpacker.exe", "Unpacker"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def run_unpacker(executable: str, source: Path, output: Path) -> None:
    completed = subprocess.run([executable, "--recursive", str(source), str(output)],
                               text=True, encoding="utf-8", errors="replace",
                               capture_output=True)
    if completed.returncode != 0:
        die(f"Falha ao extrair o InstallShield com Unpacker:\n"
            f"{completed.stdout}\n{completed.stderr}")


def read_installshield_v9_records(header_path: Path) -> list[tuple[int, bytes]]:
    """Read size/MD5 pairs from the Sonic R InstallShield 9 header.

    The file names in this particular installer are intentionally obfuscated,
    while each payload's MD5 is present in its cabinet descriptor.  MD5 is used
    here as an identifier, not as a security mechanism.
    """
    data = header_path.read_bytes()
    if len(data) < 64 or struct.unpack_from("<I", data)[0] != 0x28635349:
        die(f"Não é um cabeçalho InstallShield reconhecido: {header_path}")
    cabinet_offset = struct.unpack_from("<i", data, 12)[0]
    descriptor_offset = cabinet_offset + 12
    try:
        file_table_offset = struct.unpack_from("<i", data, descriptor_offset)[0]
        file_count = struct.unpack_from("<i", data, descriptor_offset + 28)[0]
        file_table_offset2 = struct.unpack_from("<i", data, descriptor_offset + 32)[0]
    except struct.error:
        die(f"Cabeçalho InstallShield truncado: {header_path}")
    base = cabinet_offset + file_table_offset + file_table_offset2
    if file_count < max(INSTALLER_PLAYER_TEXTURE_RECORDS.values()) + 1 or base < 0:
        die(f"Tabela de arquivos InstallShield inválida: {header_path}")
    records: list[tuple[int, bytes]] = []
    for index in range(file_count):
        offset = base + index * 0x57
        if offset + 42 > len(data):
            die(f"Tabela de arquivos InstallShield truncada: {header_path}")
        records.append((struct.unpack_from("<Q", data, offset + 2)[0], data[offset + 26:offset + 42]))
    return records


def rebuild_sonicr_data(unpacked: Path, header_path: Path, destination: Path) -> Path:
    """Restore the small named subset needed by the GLB exporter.

    Unpacker can decode the cabinet but (by design) emits its encrypted names.
    Matching its decoded bytes with the header MD5s restores only models,
    animation banks, grids and the two shared character atlases.
    """
    records = read_installshield_v9_records(header_path)
    needed = set(INSTALLER_PLAYER_TEXTURE_RECORDS.values())
    for model, grid, animations in INSTALLER_CHARACTER_RECORDS.values():
        needed.update((model, grid, *animations))
    fingerprints = {(records[index][0], records[index][1]): index for index in needed}
    located: dict[int, Path] = {}
    for candidate in unpacked.rglob("*"):
        if not candidate.is_file():
            continue
        size = candidate.stat().st_size
        digest = hashlib.md5(candidate.read_bytes()).digest()
        index = fingerprints.get((size, digest))
        if index is not None:
            located[index] = candidate
    missing = sorted(needed - located.keys())
    if missing:
        die("O descompactador não produziu todos os dados necessários do Sonic R "
            f"(registros ausentes: {missing}).")

    def copy_record(record: int, relative: Path) -> None:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(located[record], target)

    copy_record(INSTALLER_PLAYER_TEXTURE_RECORDS["PLAYER00.RAW"], Path("GENERAL") / "PLAYER00.RAW")
    copy_record(INSTALLER_PLAYER_TEXTURE_RECORDS["PLAYER01.RAW"], Path("GENERAL") / "PLAYER01.RAW")
    grid_names = {
        "sonic": "SONIC_H.GRD", "tails": "TAILS_H.GRD", "knuckles": "KNUCK_H.GRD",
        "amy": "AMY_H3.GRD", "eggman": "ROBOTNIK.GRD", "metal-sonic": "MSONIC.GRD",
        "tails-doll": "DTAILS.GRD", "metal-knuckles": "MKNUCK.GRD",
        "egg-robo": "MROBOT.GRD", "super-sonic": "SSONIC.GRD",
    }
    for spec in CHARACTERS:
        model, grid, animation_records = INSTALLER_CHARACTER_RECORDS[spec.slug]
        object_dir = Path("BIN") / "OBJECTS" / spec.object_dir
        copy_record(model, object_dir / spec.model_file)
        copy_record(grid, object_dir / grid_names[spec.slug])
        animation_names = sorted(f"{spec.animation_prefix}{number}.BIN"
                                 for number in range(1, spec.animation_count + 1))
        for name, record in zip(animation_names, animation_records, strict=True):
            copy_record(record, object_dir / name)
    return destination


def data_root_from_iso(iso: Path, stage: Path, unpacker: Path | None = None) -> Path:
    if not iso.is_file():
        die(f"ISO não encontrada: {iso}")
    disc = stage / "disc"
    run_7z(["x", "-y", str(iso), f"-o{disc}"], "extrair a ISO")
    try:
        return find_data_root(disc)
    except RuntimeError:
        pass

    setup_candidates = [entry for entry in disc.rglob("*") if entry.is_file()
                        and entry.name.casefold() == "setup.exe"]
    if not setup_candidates:
        die("A ISO não contém Setup.exe nem uma árvore de dados Sonic R reconhecível.")
    installer = stage / "installer"
    run_7z(["x", "-y", str(setup_candidates[0]), f"-o{installer}"],
           "extrair o instalador InstallShield")

    # 7-Zip can expose the bootstrapper but cannot decode this InstallShield
    # cabinet's payload.  Unpacker is an optional Windows helper; it is kept
    # outside the repository and can be supplied with --unpacker or PATH.
    tool = find_unpacker(unpacker)
    if tool:
        # The executable has an outer CAB that 7-Zip can see, followed by the
        # actual InstallShield media in its overlay.  Unpacker follows that
        # nested payload and leaves data1.hdr alongside its decoded files.
        unpacked = stage / "unpacker"
        run_unpacker(tool, setup_candidates[0], unpacked)
        header_candidates = [entry for entry in unpacked.rglob("*") if entry.is_file()
                             and entry.name.casefold() == "data1.hdr"]
        if header_candidates:
            return rebuild_sonicr_data(unpacked, header_candidates[0], stage / "data")

    # Some InstallShield versions expose embedded CABs only after the first pass.
    for archive in list(installer.rglob("*")):
        if archive.is_file() and archive.suffix.casefold() in {".cab", ".exe"}:
            target = installer / (archive.stem + "_contents")
            try:
                run_7z(["x", "-y", str(archive), f"-o{target}"],
                       f"extrair {archive.name}")
            except RuntimeError:
                # Keep scanning: a setup bootstrapper or an engine cabinet is not data.
                continue
    try:
        return find_data_root(stage)
    except RuntimeError as error:
        if tool:
            die("O InstallShield foi extraído, mas não encontrei data1.hdr para "
                f"reconstruir os nomes dos dados. Detalhe: {error}")
        die("O 7-Zip extraiu apenas o bootstrapper deste InstallShield. "
            "Instale o helper Unpacker ou passe --data-dir. "
            f"Detalhe: {error}")


def parse_model(path: Path) -> list[Limb]:
    data = path.read_bytes()
    offset = 0
    limbs: list[Limb] = []
    ordinal = 0

    def take(fmt: str) -> tuple[int, ...]:
        nonlocal offset
        size = struct.calcsize(fmt)
        if offset + size > len(data):
            die(f"Modelo truncado: {path}")
        value = struct.unpack_from(fmt, data, offset)
        offset += size
        return value

    while offset + 4 <= len(data):
        vertex_count = take("<i")[0]
        if vertex_count <= 0:
            break
        vertices: list[Vertex] = []
        for _ in range(vertex_count):
            x, y, z, nx, ny, nz, r, g, b, _pad = take("<hhhhhhBBBB")
            vertices.append(Vertex((float(x), float(y), float(z)),
                                   (float(nx), float(ny), float(nz)), (r, g, b, 255)))
        limb_index = len(limbs)
        polygons: list[Polygon] = []
        triangle_count = take("<i")[0]
        if triangle_count < 0:
            triangle_count = 0
        for _ in range(triangle_count):
            a, b, c, u0, v0, u1, v1, u2, v2, page, _pad, flags = take("<HHHBBBBBBBBh")
            if max(a, b, c) >= vertex_count:
                die(f"Índice de triângulo inválido em {path}")
            polygons.append(Polygon(limb_index, [c, b, a],
                                    [((u2 + 0.5) / 256.0, (v2 + 0.5) / 256.0),
                                     ((u1 + 0.5) / 256.0, (v1 + 0.5) / 256.0),
                                     ((u0 + 0.5) / 256.0, (v0 + 0.5) / 256.0)],
                                    1 if page else 0, ordinal, flags))
            ordinal += 1
        quad_count = take("<i")[0]
        if quad_count < 0:
            quad_count = 0
        for _ in range(quad_count):
            a, b, c, d, u0, v0, u1, v1, u2, v2, u3, v3, page, _pad, flags = take("<HHHHBBBBBBBBBBh")
            if max(a, b, c, d) >= vertex_count:
                die(f"Índice de quadrilátero inválido em {path}")
            polygons.append(Polygon(limb_index, [d, c, b, a],
                                    [((u3 + 0.5) / 256.0, (v3 + 0.5) / 256.0),
                                     ((u2 + 0.5) / 256.0, (v2 + 0.5) / 256.0),
                                     ((u1 + 0.5) / 256.0, (v1 + 0.5) / 256.0),
                                     ((u0 + 0.5) / 256.0, (v0 + 0.5) / 256.0)],
                                    1 if page else 0, ordinal, flags))
            ordinal += 1
        limbs.append(Limb(vertices, polygons))
    if not limbs:
        die(f"Nenhum membro foi encontrado no modelo {path}")
    return limbs


# The PC renderer builds Tails' two articulated tails from these vertex rings
# after drawing the normal model faces. They are not stored as polygons in
# TAILS_H.BIN, so an exporter that only reads the BIN leaves the segment sides
# open and produces the gaps visible in external viewers.
TAIL_SEGMENT_VERTICES = (
    (175, 176, 178, 179, 177, 174, 187, 181, 180, 193, 195, 188),
    (184, 185, 197, 192, 191, 194, 208, 207, 210, 209, 206, 205),
    (303, 304, 306, 308, 307, 305, 319, 318, 325, 322, 312, 311),
    (320, 321, 326, 314, 315, 323, 334, 338, 339, 337, 336, 335),
)
TAIL_FACE_MAP = ((0, 1, 7, 6), (1, 2, 8, 7), (2, 3, 9, 8),
                 (3, 4, 10, 9), (4, 5, 11, 10), (5, 0, 6, 11))


def global_vertex_refs(limbs: list[Limb]) -> dict[int, tuple[int, int]]:
    refs: dict[int, tuple[int, int]] = {}
    base = 0
    for limb_index, limb in enumerate(limbs):
        for local_index in range(len(limb.vertices)):
            refs[base + local_index] = (limb_index, local_index)
        base += len(limb.vertices)
    return refs


def apply_renderer_model_patches(spec: CharacterSpec, limbs: list[Limb]) -> list[Polygon]:
    """Apply model-load/tpage mutations performed by the original renderer."""
    polygons = [polygon for limb in limbs for polygon in limb.polygons]
    for polygon in polygons:
        logical_page = polygon.atlas
        # RemapCharacterTpages flips the logical page for all but the first
        # polygon of Amy, Tails Doll and Super Sonic. These are model-local
        # ranges because the game stores all ten models contiguously.
        if spec.char_id == 3 and 113 <= polygon.ordinal <= 116:
            logical_page = 1  # g_polyTypeTable override before the flip
        elif spec.char_id == 6 and polygon.ordinal == 0:
            logical_page = 1  # special Tails Doll first polygon
        if ((spec.char_id in {3, 6, 9}) and polygon.ordinal > 0):
            logical_page = 1 - logical_page
        polygon.atlas = logical_page

        # LoadCharacterModels rewrites Amy's vehicle polygon (global offset
        # +0x22 from Amy's model start) to the cockpit atlas rectangle and
        # marks it double-sided. Its face slots are stored D,C,B,A, like BIN
        # quads, hence the order below.
        if spec.char_id == 3 and polygon.ordinal == 34:
            polygon.uvs = [(111.5 / 256.0, 30.5 / 256.0),
                           (64.5 / 256.0, 30.5 / 256.0),
                           (64.5 / 256.0, 16.5 / 256.0),
                           (111.5 / 256.0, 16.5 / 256.0)]
            polygon.flags |= 2

    if spec.char_id != 1:
        return polygons

    refs = global_vertex_refs(limbs)
    next_ordinal = len(polygons)
    for segment, ring in enumerate(TAIL_SEGMENT_VERTICES):
        for face in TAIL_FACE_MAP:
            corner_refs = [refs[ring[index]] for index in face]
            base_u = 192 if segment % 2 == 0 else 0
            # Direct tail rendering uses A,B,C,D order (unlike BIN quads,
            # whose loader stores D,C,B,A).
            tail_uvs = [(base_u + 0.5, 128.5), (base_u + 7.5, 135.5),
                        (base_u + 7.5, 128.5), (base_u + 0.5, 135.5)]
            polygons.append(Polygon(-1, [0, 1, 2, 3],
                                    [(u / 256.0, v / 256.0) for u, v in tail_uvs],
                                    0, next_ordinal, 2, corner_refs))
            next_ordinal += 1
    return polygons


def parse_animation_bank(path: Path) -> list[tuple[int, int, int, int, int, int]]:
    data = path.read_bytes()
    if len(data) < 4 or (len(data) - 4) % 24:
        die(f"Banco de animação inválido: {path}")
    return [struct.unpack_from("<iiiiii", data, offset)
            for offset in range(4, len(data), 24)]


def parse_frame_streams(source: Path, table_name: str) -> list[list[int] | None]:
    text = source.read_text(encoding="utf-8")
    streams: dict[str, list[int]] = {}
    for name, body in re.findall(r"static short (s_fs_[0-9a-f]+)\[\]\s*=\s*\{(.*?)\};", text, re.S):
        streams[name] = [int(value) for value in re.findall(r"-?\d+", body)]
    match = re.search(rf"static uintptr_t s_ap_{re.escape(table_name)}\[\]\s*=\s*\{{(.*?)\}};", text, re.S)
    if not match:
        die(f"Tabela de animações s_ap_{table_name} não encontrada em {source}")
    # Split on entries rather than matching bare zeroes: the explanatory
    # comments beside this table also contain numeric animation IDs.
    entries: list[list[int] | None] = []
    for item in re.split(r",", match.group(1)):
        pointer = re.search(r"s_fs_[0-9a-f]+", item)
        entries.append(streams[pointer.group(0)] if pointer else None)
    if len(entries) != 18:
        die(f"Tabela s_ap_{table_name} inválida: esperava 18 entradas, recebi {len(entries)}")
    return entries


def frame_references(stream: Iterable[int]) -> list[int]:
    """Return the pose indices in a ROM stream, excluding jump operands."""
    values = list(stream)
    references: list[int] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value >= 0:
            references.append(value & 0xFFF)  # upper bits are sound triggers
        elif value in {-1, -2}:
            index += 1  # next short is the ROM-relative jump distance
        index += 1
    return references


def parse_face_states(source: Path, char_id: int) -> dict[int, dict[int, tuple[int, list[tuple[int, int]]]]]:
    """Return four {face ordinal: (atlas, four texel coordinates)} state maps."""
    text = source.read_text(encoding="utf-8")
    variants: dict[str, dict[int, tuple[int, list[tuple[int, int]]]]] = {}
    for suffix in ("A0", "B0", "A1", "B1"):
        match = re.search(rf"s_faceAnim_{char_id}_{suffix}\[\]\[10\]\s*=\s*\{{(.*?)\}};", text, re.S)
        result: dict[int, tuple[int, list[tuple[int, int]]]] = {}
        if match:
            for packed in re.findall(r"E\(([^)]*)\)", match.group(1)):
                numbers = [int(value.strip()) for value in packed.split(",")]
                if len(numbers) != 10:
                    continue
                result[numbers[0]] = (numbers[1], list(zip(numbers[2::2], numbers[3::2])))
        variants[suffix] = result
    states: dict[int, dict[int, tuple[int, list[tuple[int, int]]]]] = {}
    for flags in range(4):
        merged = dict(variants["A1" if flags & 1 else "A0"])
        merged.update(variants["B1" if flags & 2 else "B0"])
        states[flags] = merged
    return states


def png_rgb(width: int, height: int, pixels: bytes) -> bytes:
    if len(pixels) != width * height * 3:
        die("Atlas RAW não tem as dimensões RGB esperadas.")
    raw = b"".join(b"\0" + pixels[y * width * 3:(y + 1) * width * 3] for y in range(height))
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def png_rgba(width: int, height: int, pixels: bytes) -> bytes:
    """Encode an RGBA8 PNG without a third-party dependency."""
    if len(pixels) != width * height * 4:
        die("Atlas RGBA não tem as dimensões esperadas.")
    raw = b"".join(b"\0" + pixels[y * width * 4:(y + 1) * width * 4] for y in range(height))
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload +
                struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def raw_rgba(raw: bytes) -> bytes:
    """Convert Sonic R's RGB555 color-keyed atlas to RGBA8."""
    if len(raw) != 256 * 256 * 3:
        die("Atlas RAW não tem as dimensões RGB esperadas.")
    result = bytearray(256 * 256 * 4)
    for source, target in zip(range(0, len(raw), 3), range(0, len(result), 4)):
        r, g, b = raw[source:source + 3]
        # render_gl.c's IS_COLOR_KEY_RGB5: exact after quantization, not only
        # the literal (0,255,0), so all values in the 5-bit bucket are keyed.
        alpha = 0 if (r >> 3) == 0 and (g >> 3) == 31 and (b >> 3) == 0 else 255
        result[target:target + 4] = bytes((r, g, b, alpha))
    return bytes(result)


def mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[row][k] * b[k][column] for k in range(3)) for column in range(3)] for row in range(3)]


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[column][row] for column in range(3)] for row in range(3)]


def bone_matrix(rotation: tuple[int, int, int]) -> list[list[float]]:
    """Match the original limb matrix; values are 4096-angle fixed-point units."""
    x, y, z = rotation
    unit = 2.0 * math.pi / 4096.0
    sp, cp = math.sin((y + 1024) * unit), math.cos((y + 1024) * unit)
    matrix = [[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]]
    cr, sr = math.cos(z * unit), math.sin(z * unit)
    for row in matrix:
        c1, c2 = row[1], row[2]
        row[1], row[2] = c1 * cr - c2 * sr, c1 * sr + c2 * cr
    cy, sy = math.cos(x * unit), math.sin(x * unit)
    for row in matrix:
        c0, c1 = row[0], row[1]
        row[0], row[1], row[2] = c0 * cy - c1 * sy, c0 * sy + c1 * cy, -row[2]
    return matrix


BASE_BONE_MATRIX = transpose(bone_matrix((0, 0, 0)))
BASE_BONE_INVERSE = transpose(BASE_BONE_MATRIX)  # orthonormal reflection


def mat_vec(matrix: list[list[float]], value: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(sum(matrix[row][column] * value[column] for column in range(3)) for row in range(3))  # type: ignore[return-value]


def unit_vector(value: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert the game's fixed-point normal to the unit form required by glTF."""
    magnitude = math.sqrt(sum(component * component for component in value))
    if magnitude == 0:
        return (0.0, 0.0, 1.0)
    return tuple(component / magnitude for component in value)  # type: ignore[return-value]


def quat_from_matrix(matrix: list[list[float]]) -> tuple[float, float, float, float]:
    """Convert a proper row-major rotation matrix to glTF x/y/z/w quaternion."""
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = ((matrix[2][1] - matrix[1][2]) / scale, (matrix[0][2] - matrix[2][0]) / scale,
             (matrix[1][0] - matrix[0][1]) / scale, 0.25 * scale)
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        q = (0.25 * scale, (matrix[0][1] + matrix[1][0]) / scale, (matrix[0][2] + matrix[2][0]) / scale,
             (matrix[2][1] - matrix[1][2]) / scale)
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        q = ((matrix[0][1] + matrix[1][0]) / scale, 0.25 * scale, (matrix[1][2] + matrix[2][1]) / scale,
             (matrix[0][2] - matrix[2][0]) / scale)
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        q = ((matrix[0][2] + matrix[2][0]) / scale, (matrix[1][2] + matrix[2][1]) / scale, 0.25 * scale,
             (matrix[1][0] - matrix[0][1]) / scale)
    length = math.sqrt(sum(value * value for value in q))
    return tuple(value / length for value in q)  # type: ignore[return-value]


def frame_transform(frame: tuple[int, int, int, int, int, int], scale: float) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    tx, ty, tz, rx, ry, rz = frame
    current = transpose(bone_matrix((rx, ry, rz)))
    proper = mat_mul(current, BASE_BONE_INVERSE)
    return ((tx * scale, ty * scale, -tz * scale), quat_from_matrix(proper))


class BinaryBuilder:
    def __init__(self) -> None:
        self.data = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def view(self, payload: bytes, target: int | None = None) -> int:
        while len(self.data) % 4:
            self.data.append(0)
        offset = len(self.data)
        self.data.extend(payload)
        result = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
        if target is not None:
            result["target"] = target
        self.views.append(result)
        return len(self.views) - 1

    def accessor(self, payload: bytes, component_type: int, accessor_type: str, count: int,
                 target: int | None = None, minimum: list[float] | None = None,
                 maximum: list[float] | None = None, normalized: bool = False) -> int:
        view = self.view(payload, target)
        result: dict = {"bufferView": view, "componentType": component_type, "count": count, "type": accessor_type}
        if minimum is not None:
            result["min"] = minimum
        if maximum is not None:
            result["max"] = maximum
        if normalized:
            result["normalized"] = True
        self.accessors.append(result)
        return len(self.accessors) - 1


def pack_floats(values: Iterable[float]) -> bytes:
    materialized = list(values)
    return struct.pack("<" + "f" * len(materialized), *materialized)


def texture_transform(coords: list[tuple[int, int]]) -> tuple[list[float], list[float]]:
    us = [item[0] for item in coords]
    vs = [item[1] for item in coords]
    low_u, high_u = min(us), max(us)
    low_v, high_v = min(vs), max(vs)
    # The game stores these face coordinates in the same atlas convention as
    # the model UVs. Keep that convention here; flipping V changes Sonic's
    # face to a different character's cell in this packed texture.
    return [low_u / 256.0, low_v / 256.0], [(high_u - low_u) / 256.0, (high_v - low_v) / 256.0]


def default_polygon_uvs(polygon: Polygon, default_faces: dict[int, tuple[int, list[tuple[int, int]]]],
                        face_ids: set[int]) -> list[tuple[float, float]]:
    """Return the default, texel-centred UVs exactly as the game uses them."""
    if polygon.ordinal not in face_ids:
        return polygon.uvs
    _, coords = default_faces[polygon.ordinal]
    return [((u + 0.5) / 256.0, (v + 0.5) / 256.0) for u, v in coords]


def find_gouraud_file(object_root: Path) -> Path | None:
    """Find the one per-character .GRD lighting table without hardcoding names."""
    matches = sorted(path for path in object_root.iterdir() if path.is_file() and path.suffix.casefold() == ".grd")
    return matches[0] if len(matches) == 1 else None


def gouraud_row(limbs: list[Limb], path: Path, row: int = 15) -> list[list[tuple[int, int, int]]]:
    """Read one of Sonic R's 32 RGB lighting directions from a .GRD table.

    The game renderer's default global lighting phase with yaw zero selects row
    15. Values are 8-bit primary colours used by GL_ADD_SIGNED.
    """
    vertex_count = sum(len(limb.vertices) for limb in limbs)
    data = path.read_bytes()
    expected = vertex_count * 32 * 3
    if len(data) != expected:
        die(f"Tabela GRD inválida: {path} (esperava {expected} bytes, recebi {len(data)})")
    start = row * vertex_count * 3
    values = [tuple(data[start + index * 3:start + index * 3 + 3]) for index in range(vertex_count)]
    result: list[list[tuple[int, int, int]]] = []
    cursor = 0
    for limb in limbs:
        result.append(values[cursor:cursor + len(limb.vertices)])
        cursor += len(limb.vertices)
    return result


def barycentric(point: tuple[float, float], triangle: list[tuple[float, float]]) -> tuple[float, float, float]:
    (ax, ay), (bx, by), (cx, cy) = triangle
    px, py = point
    determinant = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    first = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / determinant
    second = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / determinant
    return first, second, 1.0 - first - second


def bake_add_signed_atlas(limbs: list[Limb], polygons: list[Polygon], default_faces: dict[int, tuple[int, list[tuple[int, int]]]],
                          face_ids: set[int], raw_textures: list[bytes], lighting: list[list[tuple[int, int, int]]],
                          tile_size: int = 64, gutter: int = 2) -> tuple[bytes, dict[int, list[tuple[float, float]]]]:
    """Bake Sonic R's texture + primary-colour - 0.5 combiner into one atlas.

    Character rendering uses GL_ADD_SIGNED, not the multiplicative colour
    combiner exposed by standard glTF. Each source polygon gets an independent
    tile so its Gouraud interpolation can be reproduced without Sketchfab's
    lighting model or a custom shader.
    """
    tiles_per_side = 1
    while tiles_per_side * tiles_per_side < len(polygons):
        tiles_per_side *= 2
    if tile_size <= gutter * 2:
        die("O gutter do atlas precisa deixar uma área interna positiva.")
    atlas_size = tiles_per_side * tile_size
    inner_size = tile_size - gutter * 2
    pixels = bytearray(atlas_size * atlas_size * 4)
    result_uvs: dict[int, list[tuple[float, float]]] = {}

    def source_pixel(atlas: int, u: float, v: float) -> tuple[int, int, int, int]:
        x = min(255, max(0, int(math.floor(u * 256.0))))
        y = min(255, max(0, int(math.floor(v * 256.0))))
        offset = (y * 256 + x) * 3
        r, g, b = raw_textures[atlas][offset:offset + 3]
        alpha = 0 if (r >> 3) == 0 and (g >> 3) == 31 and (b >> 3) == 0 else 255
        return r, g, b, alpha

    def polygon_corner_lights(polygon: Polygon) -> list[tuple[int, int, int]]:
        return [lighting[limb][index] for limb, index in polygon.refs()]

    for tile_index, polygon in enumerate(polygons):
        tile_x = (tile_index % tiles_per_side) * tile_size
        tile_y = (tile_index // tiles_per_side) * tile_size
        local_uvs = default_polygon_uvs(polygon, default_faces, face_ids)
        local_lights = polygon_corner_lights(polygon)
        corners = [(tile_x + gutter + 0.5, tile_y + gutter + 0.5),
                   (tile_x + tile_size - gutter - 0.5, tile_y + gutter + 0.5),
                   (tile_x + tile_size - gutter - 0.5, tile_y + tile_size - gutter - 0.5),
                   (tile_x + gutter + 0.5, tile_y + tile_size - gutter - 0.5)]
        result_uvs[polygon.ordinal] = [(x / atlas_size, y / atlas_size) for x, y in corners[:len(polygon.indices)]]
        for y in range(tile_size):
            for x in range(tile_size):
                # Extrude the rendered inner tile into its gutter. This keeps
                # nearest sampling from crossing into the unrelated polygon
                # tile next door when a viewer evaluates an edge at float
                # precision slightly outside the intended texel centre.
                inner_x = min(inner_size - 1, max(0, x - gutter))
                inner_y = min(inner_size - 1, max(0, y - gutter))
                point = ((inner_x + 0.5) / inner_size, (inner_y + 0.5) / inner_size)
                if len(polygon.indices) == 3:
                    triangle = [0, 1, 2]
                elif point[1] <= point[0]:
                    triangle = [0, 1, 2]
                else:
                    triangle = [0, 2, 3]
                weights = barycentric(point, [((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))[index] for index in triangle])
                u = sum(weights[index] * local_uvs[corner][0] for index, corner in enumerate(triangle))
                v = sum(weights[index] * local_uvs[corner][1] for index, corner in enumerate(triangle))
                texel = source_pixel(default_faces[polygon.ordinal][0] if polygon.ordinal in face_ids else polygon.atlas, u, v)
                light = [sum(weights[index] * local_lights[corner][channel] for index, corner in enumerate(triangle))
                         for channel in range(3)]
                out = ((tile_y + y) * atlas_size + tile_x + x) * 4
                for channel in range(3):
                    pixels[out + channel] = min(255, max(0, int(texel[channel] + light[channel] - 128)))
                pixels[out + 3] = texel[3]
    return png_rgba(atlas_size, atlas_size, bytes(pixels)), result_uvs


def build_glb(spec: CharacterSpec, data_root: Path, output: Path, fps: float, scale: float,
              face_variants: bool = False, vertex_colors: bool = False) -> dict:
    object_root = case_path(data_root, "BIN", "OBJECTS", spec.object_dir)
    limbs = parse_model(case_path(object_root, spec.model_file))
    frames: list[tuple[int, int, int, int, int, int]] = []
    for index in range(1, spec.animation_count + 1):
        frames.extend(parse_animation_bank(case_path(object_root, f"{spec.animation_prefix}{index}.BIN")))
    streams = parse_frame_streams(SCRIPT_ROOT / "source" / "sdl" / "src" / "anim_data.c", spec.animation_table)
    face_states = parse_face_states(SCRIPT_ROOT / "source" / "sdl" / "src" / "animation.c", spec.char_id)
    all_polygons = apply_renderer_model_patches(spec, limbs)
    face_ids = {ordinal for state in face_states.values() for ordinal in state}.intersection({poly.ordinal for poly in all_polygons})
    default_faces = face_states[0]

    raw_textures = [case_path(data_root, "GENERAL", "PLAYER00.RAW").read_bytes(),
                    case_path(data_root, "GENERAL", "PLAYER01.RAW").read_bytes()]
    pngs = [png_rgba(256, 256, raw_rgba(texture)) for texture in raw_textures]
    baked_uvs: dict[int, list[tuple[float, float]]] = {}
    # Standard GLB material colour is multiplicative, while Sonic R uses
    # GL_ADD_SIGNED. Bake the default face state with the source .GRD row so
    # it displays consistently in Sketchfab and other normal glTF viewers.
    gouraud = find_gouraud_file(object_root)
    baked_texture = None
    if not face_variants and not vertex_colors and gouraud is not None:
        baked_texture, baked_uvs = bake_add_signed_atlas(
            limbs, all_polygons, default_faces, face_ids, raw_textures, gouraud_row(limbs, gouraud))
        pngs.append(baked_texture)
    # Keep both original atlases embedded for inspection/reuse, but do not
    # create unused glTF texture objects in the normal baked export. The
    # Khronos validator correctly warns about unused textures even though the
    # corresponding images are intentionally retained inside the GLB.
    texture_sources = [2] if baked_texture is not None else [0, 1]
    texture_indices = {source: index for index, source in enumerate(texture_sources)}
    binary = BinaryBuilder()
    image_views = [binary.view(png) for png in pngs]
    document: dict = {
        "asset": {"version": "2.0", "generator": "Sonic R reusable character exporter"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "buffers": [{"byteLength": 0}],
        "bufferViews": binary.views,
        "accessors": binary.accessors,
        "images": [{"bufferView": view, "mimeType": "image/png",
                    "name": (f"PLAYER0{index}" if index < 2 else f"{spec.slug}_add_signed")}
                   for index, view in enumerate(image_views)],
        "samplers": [{"magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33071}],
        "textures": [{"source": source, "sampler": 0} for source in texture_sources],
        "materials": [], "meshes": [], "nodes": [], "skins": [], "animations": [],
        "scenes": [{"nodes": []}], "scene": 0,
    }

    def add_material(name: str, atlas: int, transform: tuple[list[float], list[float]] | None = None,
                     double_sided: bool = False) -> int:
        texinfo: dict = {"index": texture_indices[atlas]}
        if transform:
            texinfo["extensions"] = {"KHR_texture_transform": {"offset": transform[0], "scale": transform[1]}}
        document["materials"].append({"name": name, "pbrMetallicRoughness": {"baseColorTexture": texinfo,
                                      "metallicFactor": 0.0, "roughnessFactor": 1.0},
                                      "alphaMode": "MASK", "alphaCutoff": 0.5,
                                      "doubleSided": double_sided,
                                      "extensions": {"KHR_materials_unlit": {}}})
        return len(document["materials"]) - 1

    # Some characters (including Sonic) have body polygons only on PLAYER00;
    # PLAYER01 is still embedded for face-animation states.  Do not create an
    # unused base material merely to hold that second texture.
    used_body_atlases = {polygon.atlas for polygon in all_polygons}
    baked_materials = ({side: add_material(f"{spec.slug}_add_signed_{'double' if side else 'cull'}", 2,
                                           double_sided=side) for side in (False, True)}
                       if baked_texture is not None else {})
    base_materials = {} if baked_texture is not None else {
        (atlas, side): add_material(f"PLAYER0{atlas}_{'double' if side else 'cull'}", atlas,
                                    double_sided=side)
        for atlas in sorted(used_body_atlases) for side in (False, True)
    }
    polygon_by_ordinal = {polygon.ordinal: polygon for polygon in all_polygons}
    face_materials: dict[int, int] = {}
    if face_variants:
        for face_id in sorted(face_ids):
            atlas, coords = default_faces.get(face_id, (0, [(0, 0), (255, 0), (255, 255), (0, 255)]))
            face_materials[face_id] = add_material(
                f"face_{face_id:03d}", atlas, texture_transform(coords),
                polygon_by_ordinal[face_id].double_sided)
    elif baked_texture is None:
        used_face_atlases = {default_faces[face_id][0] for face_id in face_ids if face_id in default_faces}
        for atlas in used_face_atlases:
            for side in (False, True):
                base_materials.setdefault((atlas, side), add_material(
                    f"PLAYER0{atlas}_{'double' if side else 'cull'}", atlas, double_sided=side))
    if face_materials:
        document["extensionsUsed"].append("KHR_texture_transform")
        if face_variants:
            document["extensionsUsed"].append("KHR_animation_pointer")

    # The game overwrites its base vertex colours from a direction-dependent
    # .GRD lighting table immediately before rendering. The default output
    # bakes that add-signed result into a texture; exporting the bootstrap
    # values as COLOR_0 would instead make unlit glTF viewers show false
    # lighting patches and triangle seams.
    # --vertex-colors retains the raw values for diagnostic use, but never
    # applies them to face patches.
    #
    # Face polygons must therefore be distinct primitives even if they sample
    # the same atlas/material as adjacent body polygons.
    groups: dict[tuple[int, int, bool], list[Polygon]] = {}
    for polygon in all_polygons:
        if baked_texture is not None:
            material = baked_materials[polygon.double_sided]
        elif polygon.ordinal in face_materials:
            material = face_materials[polygon.ordinal]
        elif polygon.ordinal in default_faces:
            material = base_materials[(default_faces[polygon.ordinal][0], polygon.double_sided)]
        else:
            material = base_materials[(polygon.atlas, polygon.double_sided)]
        groups.setdefault((polygon.limb, material, polygon.ordinal in face_ids), []).append(polygon)
    primitives: list[dict] = []
    for (limb_index, material, is_face_primitive), polygons in groups.items():
        positions: list[float] = []
        normals: list[float] = []
        texcoords: list[float] = []
        colors = bytearray()
        joints: list[int] = []
        weights: list[float] = []
        indices: list[int] = []
        for polygon in polygons:
            override = default_faces.get(polygon.ordinal)
            is_face = override is not None and polygon.ordinal in face_ids
            refs = polygon.refs()
            face_uvs: list[tuple[float, float]] | None = None
            if is_face:
                _, coords = override
                if face_variants:
                    low_u, high_u = min(point[0] for point in coords), max(point[0] for point in coords)
                    low_v, high_v = min(point[1] for point in coords), max(point[1] for point in coords)
                    width, height = max(1, high_u - low_u), max(1, high_v - low_v)
                    face_uvs = [((u - low_u) / width, (v - low_v) / height) for u, v in coords]
                else:
                    face_uvs = default_polygon_uvs(polygon, default_faces, face_ids)
            start = len(positions) // 3
            for local_index, (vertex_limb, vertex_index) in enumerate(refs):
                vertex = limbs[vertex_limb].vertices[vertex_index]
                p = mat_vec(BASE_BONE_MATRIX, vertex.position)
                n = unit_vector(mat_vec(BASE_BONE_MATRIX, vertex.normal))
                positions.extend((p[0] * scale, p[1] * scale, p[2] * scale))
                normals.extend(n)
                uv = (baked_uvs[polygon.ordinal][local_index] if baked_texture is not None
                      else face_uvs[local_index] if face_uvs and local_index < len(face_uvs)
                      else polygon.uvs[local_index])
                texcoords.extend(uv)
                if vertex_colors and not is_face_primitive:
                    colors.extend(vertex.color)
                joints.extend((vertex_limb, 0, 0, 0))
                weights.extend((1.0, 0.0, 0.0, 0.0))
            indices.extend((start, start + 1, start + 2))
            if len(refs) == 4:
                indices.extend((start, start + 2, start + 3))
        if len(positions) // 3 > 65535:
            index_payload, index_component = struct.pack("<" + "I" * len(indices), *indices), 5125
        else:
            index_payload, index_component = struct.pack("<" + "H" * len(indices), *indices), 5123
        position_values = list(zip(positions[0::3], positions[1::3], positions[2::3]))
        attributes = {
            "POSITION": binary.accessor(pack_floats(positions), 5126, "VEC3", len(position_values), 34962,
                                        [min(v[i] for v in position_values) for i in range(3)], [max(v[i] for v in position_values) for i in range(3)]),
            "NORMAL": binary.accessor(pack_floats(normals), 5126, "VEC3", len(position_values), 34962),
            "TEXCOORD_0": binary.accessor(pack_floats(texcoords), 5126, "VEC2", len(position_values), 34962),
            "JOINTS_0": binary.accessor(struct.pack("<" + "H" * len(joints), *joints), 5123, "VEC4", len(position_values), 34962),
            "WEIGHTS_0": binary.accessor(pack_floats(weights), 5126, "VEC4", len(position_values), 34962),
        }
        if vertex_colors and not is_face_primitive:
            attributes["COLOR_0"] = binary.accessor(bytes(colors), 5121, "VEC4", len(position_values), 34962,
                                                       normalized=True)
        primitive = {"attributes": attributes,
                     "indices": binary.accessor(index_payload, index_component, "SCALAR", len(indices), 34963),
                     "material": material}
        primitives.append(primitive)
    document["meshes"].append({"name": spec.display_name, "primitives": primitives})

    # Default pose is the first idle frame.  The per-limb base reflection is
    # baked into vertices above, so these joint values are valid quaternions.
    idle_refs = frame_references(streams[0] or [])
    if not idle_refs:
        die(f"A tabela de idle de {spec.display_name} está vazia.")
    default_frame_index = idle_refs[0] - 1
    if default_frame_index < 0 or default_frame_index * len(limbs) + len(limbs) > len(frames):
        die(f"Frame inicial de {spec.display_name} está fora dos bancos de animação.")
    root_node = len(document["nodes"])
    document["nodes"].append({"name": f"{spec.display_name}_Skeleton", "children": []})
    joint_nodes: list[int] = []
    for limb_index in range(len(limbs)):
        transform = frame_transform(frames[default_frame_index * len(limbs) + limb_index], scale)
        joint_nodes.append(len(document["nodes"]))
        document["nodes"].append({"name": f"{spec.display_name}_Limb_{limb_index:02d}", "translation": transform[0], "rotation": transform[1]})
    document["nodes"][root_node]["children"] = joint_nodes
    identity_matrices = [value for _ in limbs for value in (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)]
    skin = {"name": f"{spec.display_name}_RigidSkin", "joints": joint_nodes, "skeleton": root_node,
            "inverseBindMatrices": binary.accessor(pack_floats(identity_matrices), 5126, "MAT4", len(limbs))}
    document["skins"].append(skin)
    mesh_node = len(document["nodes"])
    document["nodes"].append({"name": spec.display_name, "mesh": 0, "skin": 0})
    document["scenes"][0]["nodes"] = [root_node, mesh_node]

    def add_animation(name: str, references: list[int], face_flags: int | None) -> None:
        if not references:
            return
        if any(ref <= 0 or ref * len(limbs) > len(frames) for ref in references):
            die(f"{name}: referência de frame fora dos bancos de animação.")
        times = [index / fps for index in range(len(references))]
        input_accessor = binary.accessor(pack_floats(times), 5126, "SCALAR", len(times), None, [times[0]], [times[-1]])
        animation: dict = {"name": name, "samplers": [], "channels": []}
        for limb_index, node in enumerate(joint_nodes):
            translations: list[float] = []
            rotations: list[float] = []
            for reference in references:
                translation, rotation = frame_transform(frames[(reference - 1) * len(limbs) + limb_index], scale)
                translations.extend(translation)
                rotations.extend(rotation)
            translation_accessor = binary.accessor(pack_floats(translations), 5126, "VEC3", len(references))
            rotation_accessor = binary.accessor(pack_floats(rotations), 5126, "VEC4", len(references))
            sampler = len(animation["samplers"])
            animation["samplers"].append({"input": input_accessor, "output": translation_accessor, "interpolation": "STEP"})
            animation["channels"].append({"sampler": sampler, "target": {"node": node, "path": "translation"}})
            sampler = len(animation["samplers"])
            animation["samplers"].append({"input": input_accessor, "output": rotation_accessor, "interpolation": "STEP"})
            animation["channels"].append({"sampler": sampler, "target": {"node": node, "path": "rotation"}})
        if face_flags is not None:
            for face_id, material in face_materials.items():
                atlas, coords = face_states[face_flags].get(face_id, default_faces[face_id])
                offset, extent = texture_transform(coords)
                for suffix, values, accessor_type in (("offset", offset, "VEC2"), ("scale", extent, "VEC2"), ("index", [float(atlas)], "SCALAR")):
                    output_values = values * len(references)
                    output_accessor = binary.accessor(pack_floats(output_values), 5126, accessor_type, len(references))
                    sampler = len(animation["samplers"])
                    animation["samplers"].append({"input": input_accessor, "output": output_accessor, "interpolation": "STEP"})
                    if suffix == "index":
                        pointer = f"/materials/{material}/pbrMetallicRoughness/baseColorTexture/index"
                    else:
                        pointer = f"/materials/{material}/pbrMetallicRoughness/baseColorTexture/extensions/KHR_texture_transform/{suffix}"
                    animation["channels"].append({"sampler": sampler, "target": {"path": "pointer", "extensions": {"KHR_animation_pointer": {"pointer": pointer}}}})
        document["animations"].append(animation)

    for animation_id, stream in enumerate(streams):
        if not stream:
            continue
        refs = frame_references(stream)
        base_name = ANIMATION_NAMES.get(animation_id, f"animation_{animation_id}")
        if face_materials and face_variants:
            for flags in range(4):
                add_animation(f"{base_name}_face{flags:02b}", refs, flags)
        else:
            add_animation(base_name, refs, None)

    document["buffers"][0]["byteLength"] = len(binary.data)
    document["bufferViews"] = binary.views
    document["accessors"] = binary.accessors
    output.parent.mkdir(parents=True, exist_ok=True)
    json_chunk = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    while len(json_chunk) % 4:
        json_chunk += b" "
    bin_chunk = bytes(binary.data)
    while len(bin_chunk) % 4:
        bin_chunk += b"\0"
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    output.write_bytes(struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk + struct.pack("<I4s", len(bin_chunk), b"BIN\0") + bin_chunk)
    return validate_glb(output)


def validate_glb(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) < 20:
        die(f"GLB pequeno demais: {path}")
    magic, version, length = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2 or length != len(data):
        die(f"Cabeçalho GLB inválido: {path}")
    json_length, chunk_type = struct.unpack_from("<I4s", data, 12)
    if chunk_type != b"JSON":
        die(f"JSON ausente no GLB: {path}")
    document = json.loads(data[20:20 + json_length])
    if document.get("buffers", [{}])[0].get("byteLength", 0) > len(data):
        die(f"Buffer GLB inválido: {path}")
    if len(document.get("images", [])) < 2 or any("uri" in image for image in document["images"]):
        die(f"O GLB deve conter os dois atlas incorporados: {path}")
    if not document.get("skins") or not document.get("animations"):
        die(f"Skin ou animações ausentes no GLB: {path}")
    return {"animations": len(document["animations"]), "joints": len(document["skins"][0]["joints"]),
            "images": len(document["images"]), "extensions": document.get("extensionsUsed", [])}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--iso", type=Path, help="ISO Sonic R para extrair temporariamente")
    source.add_argument("--data-dir", type=Path, help="Pasta já extraída/instalada que contém GENERAL e BIN")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--character", choices=sorted(CHARACTER_BY_SLUG), help="Personagem a exportar")
    selection.add_argument("--all", action="store_true", help="Exportar todos os personagens")
    parser.add_argument("--output", type=Path, required=True, help="Arquivo .glb, ou pasta ao usar --all")
    parser.add_argument("--fps", type=float, default=FPS_DEFAULT,
                        help="Ticks de animação por segundo (padrão fiel ao jogo: 30)")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="Escala das unidades do jogo (padrão: 1/16)")
    parser.add_argument("--unpacker", type=Path,
                        help="Unpacker.exe para cabinets InstallShield (opcional)")
    parser.add_argument("--face-variants", action="store_true",
                        help="Exportar quatro variantes faciais por clipe (68 no Sonic; requer KHR_animation_pointer)")
    parser.add_argument("--vertex-colors", action="store_true",
                        help="Manter as cores-base dos vértices (diagnóstico; pode criar iluminação estática)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.fps <= 0 or args.scale <= 0:
        die("--fps e --scale devem ser positivos.")
    # Keep ISO staging on the same volume as the requested output. This avoids
    # silently consuming the system C: drive for a several-hundred-megabyte ISO.
    output_base = args.output if args.all else args.output.parent
    output_base.mkdir(parents=True, exist_ok=True)
    if args.iso:
        with tempfile.TemporaryDirectory(prefix=".sonic-r-stage-", dir=output_base) as temporary:
            root = data_root_from_iso(args.iso, Path(temporary), args.unpacker)
            characters = CHARACTERS if args.all else (CHARACTER_BY_SLUG[args.character],)
            for spec in characters:
                output = (args.output / f"{spec.slug}.glb") if args.all else args.output
                if not args.all and output.suffix.casefold() != ".glb":
                    die("--output deve terminar em .glb, ou ser uma pasta ao usar --all.")
                report = build_glb(spec, root, output, args.fps, args.scale, args.face_variants, args.vertex_colors)
                print(f"Exportado {spec.display_name}: {output} ({report['joints']} joints, {report['animations']} animações)")
    else:
        root = find_data_root(args.data_dir)
        characters = CHARACTERS if args.all else (CHARACTER_BY_SLUG[args.character],)
        for spec in characters:
            output = (args.output / f"{spec.slug}.glb") if args.all else args.output
            if not args.all and output.suffix.casefold() != ".glb":
                die("--output deve terminar em .glb, ou ser uma pasta ao usar --all.")
            report = build_glb(spec, root, output, args.fps, args.scale, args.face_variants, args.vertex_colors)
            print(f"Exportado {spec.display_name}: {output} ({report['joints']} joints, {report['animations']} animações)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"erro: {error}", file=sys.stderr)
        raise SystemExit(2)
