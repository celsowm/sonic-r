# Sonic R

Here you will find a reimplementation of *Sonic R* (the 1998 Sega PC release),
decompiled from the original executable and rebuilt as modern, portable C.
It runs natively on **Windows, macOS, and Linux** (via SDL2) and on the **Sega Dreamcast**
(via KallistiOS), sharing one game-logic codebase across all targets.

## AI Disclosure
This project is the product of several months of reverse-engineering and 
x86-to-C translation done with the use of Ghidra and Claude Code
(using Opus 4.6, Opus 4.8 and Fable 5). 

This has been extensively tested and validated against live WinDbg sessions running 1998 SONICR.EXE,
along with playtesting by many *detail-oriented* Sonic R fans (special thanks to Mittens, Tongara and Neo.charmy).

We ran out of identifiable differences in gameplay behavior that could be attributed to anything other than switching from 80-bit x87 floating point to IEEE floating point math (or using modified game data).

If this use of AI is in conflict with your personal beliefs or values, please leave this page now. Thanks.

## Play

> **You supply your own game data.** This port is **only the game engine** — it
> does **not** include Sonic R's assets. You must own a copy of the game.

- **▶ Windows, macOS & Linux** → **[docs/desktop.md](docs/desktop.md)**
- **▶ Sega Dreamcast** → **[docs/dreamcast.md](docs/dreamcast.md)**

## Exporting character GLBs

`tools/export_sonicr_character.py` converts user-supplied Sonic R PC data into
a self-contained, rigidly skinned GLB with embedded character atlases and game
animation clips. It supports all ten characters, not just Sonic. Python 3.12+
is sufficient; ISO input also needs `7z` available on `PATH`.

```powershell
python tools/export_sonicr_character.py --iso E:\SONICR.ISO --unpacker E:\tools\Unpacker.exe --character sonic --output exports\sonic.glb
python tools/export_sonicr_character.py --data-dir D:\Games\SonicR --all --output exports
python tools/export_sonicr_character.py --data-dir D:\Games\SonicR --character sonic --face-variants --output exports\sonic-face-variants.glb
```

The exporter never commits or retains extracted game data. Sonic R's
InstallShield installer cannot be decoded by 7-Zip alone. Pass the optional
`Unpacker.exe` helper with `--unpacker` (or put it on `PATH`); the exporter
uses the cabinet's embedded checksums to restore the model, animation and atlas
file names without running the installer. `--data-dir` remains available for
an already extracted or installed copy. The standard Sonic export has the 17
named body clips; `--face-variants` opts into the 68 face-atlas variants for
viewers that support `KHR_animation_pointer`.
For normal exports, the tool also embeds a third, generated atlas that bakes
Sonic R's `GL_ADD_SIGNED` texture combiner with the default `.GRD` lighting
row. This is necessary because standard glTF can multiply vertex colours but
cannot express the game's add-signed combiner; the baked/unlit material keeps
the result stable in Sketchfab. `--vertex-colors` is diagnostic-only. Face
variant exports retain the original texture-animation path instead of this
static bake.

Each guide is self-contained: how to get or build it, the game data you supply,
controls, network play, saves, and troubleshooting for that platform.

## Legal & credits

I personally report every for-sale listing of my ports that are brought to my attention. I will get your 15 year old 100% feedback eBay account terminated.

Sonic R and its characters, trademarks, and game assets are the property of
Sega. This is a from-scratch reimplementation of the game engine — it contains
no original Sega code — and you must own a copy of Sonic R to supply the track,
model, texture, and music data needed to play.

A few small supporting assets are bundled under `DATA/` for convenience: the
custom network player platform icons (`NET01.RAW`) and the pre-encoded sound fx.
Everything else must come from your own copy of the game.

This is a non-commercial preservation project and is not affiliated with or
endorsed by Sega.

If you sell this, I hope all the bad things in life happen to you and only to you.
