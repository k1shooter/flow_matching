# Music Note-Tuple Discrete Flow Matching (DFM + Optional EditFlow)

This example extends `examples/text` to symbolic music generation with note tuples:

`x_i = (track, bar, pos, pitch, dur, vel)`

- All fields are categorical.
- `PAD=0`, non-PAD categories start from `1`.
- Default representation is fixed-length slots (`max_notes_per_segment`).
- Optional EditFlow mode enables variable-length generation.

## 1) Environment

```bash
cd examples/music_note_tuple
conda env create -f environment.yml
conda activate discrete_flow_matching
```

If you already use another env (e.g. `flow_matching`), install required packages there:

```bash
pip install hydra-core hydra-submitit-launcher omegaconf pretty_midi music21 pyfluidsynth
```

Optional score image backend (recommended):

```bash
pip install cairosvg
pip install verovio  # may require system swig/cmake toolchain
```

If `verovio` build fails, you can skip it and use MuseScore CLI fallback:

```bash
sudo apt-get update
sudo apt-get install -y musescore3
```

Important: install `hydra-core`, not `hydra`.

Default run outputs are saved under `./runs` (inside the current project directory).

## 2) Training

### 2.1 Base DFM (fixed-length)

```bash
python run_train.py \
  hydra_dir=/abs/path/to/runs \
  flow.use_edit_flow=false
```

- `data.midi_dir`를 명시하지 않으면 학습 시작 시 `data.auto_download.url`에서 dataset archive를 자동 다운로드/압축해제합니다.
- 이후 MIDI -> note-tuple 전처리를 split(train/valid)별 `.pt` 캐시로 저장하고, 재실행 시 캐시를 재사용합니다.
- 기본 캐시 경로:
  - dataset archive/extract: `~/.cache/flow_matching_music_note_tuple/datasets`
  - preprocessed tuples: `~/.cache/flow_matching_music_note_tuple/preprocessed`

직접 데이터셋 경로를 지정하려면:

```bash
python run_train.py \
  data.midi_dir=/abs/path/to/midi/root \
  hydra_dir=/abs/path/to/runs \
  flow.use_edit_flow=false
```

### 2.2 Optional EditFlow mode (variable-length sampler)

```bash
python run_train.py \
  hydra_dir=/abs/path/to/runs \
  flow.use_edit_flow=true \
  flow.edit_flow.sampler=tau_leaping
```

EditFlow knobs:
- `flow.edit_flow.sampler`: `tau_leaping` or `gillespie`
- `flow.edit_flow.tau_leaping_step_size`
- `flow.edit_flow.max_events_per_step`
- `flow.edit_flow.rate_scale`
- `flow.edit_flow.init_mode`: `empty` or `uniform`

Auto download / cache 관련 주요 오버라이드:
- `data.auto_download.url=...` (직접 링크 변경)
- `data.auto_download.archive_name=...`
- `data.auto_download.midi_subdir=...` (압축 내부 특정 MIDI 폴더를 루트로 사용)
- `data.auto_download.force_download=true`
- `data.auto_download.force_extract=true`
- `data.rebuild_preprocess_cache=true` (전처리 캐시 강제 재생성)
- `data.preprocess_cache_version=2` (버전 증가로 새 캐시 키 생성)

Training preview generation (default enabled):
- epoch 0 시작 시(`training.preview.sample_at_start=true`) 랜덤 source 샘플을 먼저 생성합니다.
- every `training.preview.every_steps`, a few samples are generated during training
- outputs are saved under `<run_dir>/samples/step_XXXXXXXX/` as MIDI + MusicXML + score PNG + WAV
- control with `training.preview.*` in config
- preview objective evaluation is also run and saved under `<run_dir>/samples/step_XXXXXXXX/eval/`
  using `training.preview.eval.*` (default protocol follows `eval.eval_protocol`)
- if `fluidsynth: warning: SDL2 not initialized ...` appeared before, WAV rendering now uses
  headless-safe settings and stderr suppression in the renderer.
- preview audio is piano-only by default (`training.preview.piano_only=true`).
  If you want multi-instrument timbre, set `training.preview.piano_only=false`
  and customize `data.track_programs`.

## 3) Sampling

```bash
PYTHONPATH="." python scripts/sample_and_save_midi.py \
  --work_dir /abs/path/to/run \
  --output_dir /abs/path/to/generated_midis \
  --num_samples 16 \
  --batch_size 8 \
  --seed 42
```

- If the trained config has `flow.use_edit_flow=true`, sampling automatically uses EditFlow variable-length updates.
- Otherwise it uses standard DFM masked-resample updates.
- By default, outputs are saved as:
- `midi/*.mid`
- `musicxml/*.musicxml` (score file)
- `score_png/*.png` (sheet image)
- `wav/*.wav` (audio rendering)

Useful options:
- `--disable_musicxml`
- `--disable_score_png`
- `--disable_wav`
- `--soundfont_path /abs/path/to/sf2`
- `--score_png_scale 45`
- WAV rendering needs `pyfluidsynth` + system FluidSynth + a valid `.sf2` soundfont.
- score PNG rendering prefers `verovio` + `cairosvg`; fallback is headless `musescore3` CLI.

If WAV export still fails with preview warning, install system packages (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y fluidsynth fluid-soundfont-gm
```

Then set soundfont path in training config/CLI, e.g.:

```bash
python run_train.py training.preview.soundfont_path=/usr/share/sounds/sf2/FluidR3_GM.sf2
```

Quick checks:

```bash
which fluidsynth
ls /usr/share/sounds/sf2/FluidR3_GM.sf2
```

## 4) Evaluation (single command)

### 4.1 D3PIA POP909 protocol

```bash
PYTHONPATH="." python scripts/eval.py \
  --protocol d3pia_pop909 \
  --ckpt /abs/path/to/run_or_checkpoint \
  --reference_dir /abs/path/to/pop909_test_midis \
  --out_dir /abs/path/to/eval_out \
  --key_labels_csv /abs/path/to/pop909_key_labels.csv
```

### 4.2 WholeSongGen ICLR24 protocol

```bash
PYTHONPATH="." python scripts/eval.py \
  --protocol wholesonggen_iclr24 \
  --ckpt /abs/path/to/run_or_checkpoint \
  --train_dir /abs/path/to/train_midis \
  --out_dir /abs/path/to/eval_out
```

Both protocols save:
- JSON (`objective_summary.json` / protocol summary JSON)
- CSV (segment-level or metric-level)
- Markdown table (`table_*.md`)

## 5) Subjective evaluation package automation

### D3PIA package creation

```bash
PYTHONPATH="." python scripts/eval.py \
  --protocol d3pia_pop909 \
  --generated_dir /abs/path/to/generated_midis \
  --reference_dir /abs/path/to/pop909_test_midis \
  --out_dir /abs/path/to/eval_out \
  --create_subjective_package
```

### WholeSongGen package creation

```bash
PYTHONPATH="." python scripts/eval.py \
  --protocol wholesonggen_iclr24 \
  --generated_dir /abs/path/to/generated_midis \
  --train_dir /abs/path/to/train_midis \
  --out_dir /abs/path/to/eval_out \
  --create_subjective_package
```

Then analyze survey CSV:

```bash
PYTHONPATH="." python scripts/eval.py \
  --protocol d3pia_pop909 \
  --generated_dir /abs/path/to/generated_midis \
  --reference_dir /abs/path/to/pop909_test_midis \
  --out_dir /abs/path/to/eval_out \
  --subjective_csv /abs/path/to/filled_survey.csv
```

(or with `--protocol wholesonggen_iclr24` for whole-song subjective analysis)

## 6) Protocol notes and reproducibility

Implemented protocol outputs:
- D3PIA-like objective bundle: `OOK`, `CA`, `CS`, `GS`
- WholeSongGen-like objective bundle: `ILS_{p,r,chd,txt}`, `DoC_{simlt_p,simlt_r,simrb}`
- Subjective package + CSV templates + summary/CI analysis

### Important comparability notes

To exactly match paper numbers, these external assets/conventions must match the original setup:
- POP909 official split metadata and key/chord labels
- Pretrained chord encoder checkpoint used by D3PIA
- Pretrained latent encoders used by WholeSongGen

If checkpoints/labels are missing, the code runs with deterministic fallbacks and logs a comparability risk message.

## 7) Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Included sanity tests cover:
- note tuple <-> pianoroll roundtrip
- OOK/GS correctness on toy data
- ILS sanity behavior
- DoC `S(x)` sanity
- EditFlow length-change behavior
- preprocessing cache reuse behavior
