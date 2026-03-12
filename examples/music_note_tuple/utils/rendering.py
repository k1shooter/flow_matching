# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import warnings

from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def _suppress_stderr_fd():
    """Suppress C-library stderr output (e.g. libfluidsynth warnings) temporarily."""
    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        yield
        return

    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        try:
            os.dup2(saved_stderr_fd, stderr_fd)
        finally:
            os.close(saved_stderr_fd)


def maybe_render_midi_to_wav(
    midi_path: Path,
    wav_path: Path,
    soundfont_path: str | None = None,
    sample_rate: int = 44_100,
) -> bool:
    try:
        import pretty_midi
    except Exception:
        return False

    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        # Use dummy SDL audio backend to avoid initialization warnings in headless envs.
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        with _suppress_stderr_fd():
            if soundfont_path is not None:
                audio = pm.fluidsynth(fs=sample_rate, sf2_path=soundfont_path)
            else:
                audio = pm.fluidsynth(fs=sample_rate)

        import wave

        wav_path.parent.mkdir(parents=True, exist_ok=True)
        audio16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
        with wave.open(str(wav_path), "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio16.tobytes())
        return True
    except Exception:
        return False


def maybe_render_midi_to_musicxml(midi_path: Path, musicxml_path: Path) -> bool:
    try:
        from music21 import converter
    except Exception:
        return False

    try:
        score = converter.parse(str(midi_path))
        musicxml_path.parent.mkdir(parents=True, exist_ok=True)
        score.write("musicxml", fp=str(musicxml_path))
        return True
    except Exception:
        return False


def _fallback_render_musicxml_to_png_with_music21(
    musicxml_path: Path, png_path: Path
) -> bool:
    try:
        from music21 import converter
    except Exception:
        return False

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = converter.parse(str(musicxml_path))
        musescore_bin = _find_musescore_binary()
        if musescore_bin is not None:
            try:
                from music21 import environment

                user_settings = environment.UserSettings()
                user_settings["musescoreDirectPNGPath"] = musescore_bin
            except Exception:
                pass
        png_path.parent.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out_path = score.write("musicxml.png", fp=str(png_path))
        candidate = Path(out_path) if out_path is not None else png_path
        if candidate.exists() and candidate != png_path:
            if candidate.suffix.lower() == ".png":
                shutil.copy2(candidate, png_path)
            elif (
                candidate.suffix.lower() == ".musicxml"
                and candidate.resolve() != musicxml_path.resolve()
            ):
                # music21 can create an intermediate xml when direct PNG export fails.
                candidate.unlink(missing_ok=True)
        return png_path.exists()
    except Exception:
        return False


def _find_musescore_binary() -> str | None:
    for candidate in ("musescore3", "mscore", "musescore"):
        binary = shutil.which(candidate)
        if binary:
            return binary
    return None


def _fallback_render_musicxml_to_png_with_musescore_cli(
    musicxml_path: Path, png_path: Path
) -> bool:
    musescore_bin = _find_musescore_binary()
    if musescore_bin is None:
        return False

    png_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="musicxml_png_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        output_prefix = tmp_dir / "score"
        output_path = output_prefix.with_suffix(".png")

        env = os.environ.copy()
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        runtime_dir = tmp_dir / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(runtime_dir, 0o700)
        env.setdefault("XDG_RUNTIME_DIR", str(runtime_dir))

        try:
            subprocess.run(
                [musescore_bin, "-o", str(output_path), str(musicxml_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
        except Exception:
            return False

        candidates = []
        if output_path.exists():
            candidates.append(output_path)
        candidates.extend(sorted(tmp_dir.glob("score-*.png")))
        if not candidates:
            return False

        shutil.copy2(candidates[0], png_path)
        return png_path.exists()


def maybe_render_musicxml_to_png(
    musicxml_path: Path,
    png_path: Path,
    scale: int = 45,
) -> bool:
    try:
        import verovio
        from cairosvg import svg2png
    except Exception:
        if _fallback_render_musicxml_to_png_with_musescore_cli(
            musicxml_path=musicxml_path,
            png_path=png_path,
        ):
            return True
        return _fallback_render_musicxml_to_png_with_music21(
            musicxml_path=musicxml_path,
            png_path=png_path,
        )

    try:
        tk = verovio.toolkit()
        tk.setOptions(
            {
                "scale": int(scale),
                "pageWidth": 2100,
                "pageHeight": 2970,
                "footer": "none",
                "header": "none",
            }
        )
        loaded = tk.loadFile(str(musicxml_path))
        if loaded is False:
            return False

        svg = tk.renderToSVG(1)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        svg2png(bytestring=svg.encode("utf-8"), write_to=str(png_path))
        return png_path.exists()
    except Exception:
        if _fallback_render_musicxml_to_png_with_musescore_cli(
            musicxml_path=musicxml_path,
            png_path=png_path,
        ):
            return True
        return _fallback_render_musicxml_to_png_with_music21(
            musicxml_path=musicxml_path,
            png_path=png_path,
        )
