# fiber_sheet: fibre-guided sheet reconstruction for PHerc0846A

Trace papyrus fibres through the CT with the AFV fibre model, check which fibres lie on the same sheet by
how they cross, and turn the verified fibres into a VC3D surface (tifxyz).

The fibres are treated as the ground truth: a sheet that switches layers is corrected by forcing it onto
the fibres. Section 5 describes the whole process end to end (fibre tracing, verification, fibre-only surface,
flattening, hand correction, surface-first cleaning); sections 3 and 7 document the individual tools.

```
 x,y,z point ──► analyze_fiber.py ──► fibre (.json for VC3D, .path.npy) + crossing / layer report
                        │
 existing fibre ──► extend_fiber.py ──► longer fibre (z-locked, x-monotonic) + crossing report
                        │
 fibre folder  ──► fibers_to_tifxyz.py ──► tifxyz segment (x.tif, y.tif, z.tif, meta.json) for VC3D
```

---

## 1. Setup

### Python environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch     # pick the wheel for your CUDA
pip install -r requirements.txt
```

A CUDA GPU is needed only when the model has to predict tiles that are not cached yet. Everything else
runs on the CPU.

### Model weights

The fibre model is the **AFV fibre model V2 by Qualzz20** (Apache-2.0). It continues the
[AFV ~9 µm fibre model](https://huggingface.co/Qualzz20/afv_fiber_9um). This repository does not contain
the model code or weights; get them from the author's Hugging Face page and put them in one folder:

```
<model folder>/
  model.py
  plans.json
  dataset.json
  checkpoint_final.pth
```

By default this folder is `~/.VC3D/afv_model`. To use another location, set `AFV_MODEL_DIR`:

```powershell
$env:AFV_MODEL_DIR = "D:\models\afv_fiber_9um_v2"
```

### Data and caches

The scripts are set up for **PHerc0846A, volume 20250728152254 (9.362 µm)** from the Vesuvius Challenge
open data (the URL is in `tools/afv_core.py`). Nothing has to be downloaded by hand:

| What | Where (default) | Notes |
|---|---|---|
| CT chunks | `~/.VC3D/afv_ct_cache/20250728152254/` | Downloaded on demand, 128³ chunks, a few GB for one sheet |
| Fibre predictions | `~/.VC3D/afv_predictions/PHerc0846A_20250728152254.zarr` | `vt`/`hz` probabilities (×255) and `dir`, computed on demand per 96³ tile |

Set `VC3D_HOME` to move both caches somewhere other than `~/.VC3D`.

### A reference sheet

All three scripts take a **tifxyz segment** (a folder with `x.tif`, `y.tif`, `z.tif`, `meta.json`). Only
its orientation is used, as the layer normal (`cross(du, dv)` of its grid). It may be wrong by a layer
here and there; that is what this repository fixes. The examples below call it `$SHEET`:

```powershell
$SHEET = "C:\Users\ChenJeff\.VC3D\remote_cache\open_data\projects\paths\auto_grown_20261012065803387_straightened2_abf"
```

---

## 2. Concepts you need

- **Coordinates** are volume voxels `x, y, z` (as in VC3D). In this region horizontal fibres run along x
  and vertical fibres along z.
- **Families.** `h_…` is a horizontal fibre and `v_…` a vertical fibre. A fibre is named after the point
  it was seeded from: `h_3394_3491_13385` = horizontal seeded at x=3394, y=3491, z=13385.
- **Two plies per sheet.** Horizontal and vertical fibres of one sheet lie on opposite faces of it. The
  horizontal sits **+p ≈ 9 vx** from its own vertical along the sheet normal, and sheets repeat every
  **S ≈ 12–14 vx** (≈ 131 µm; the fit on `data/grid1_final26` gives 12.0). Both numbers are fitted from the data (`layers.json`).
- **Same-sheet rule (one-sided).** At a crossing, the offset `o` = horizontal minus vertical along the
  normal decides it:
  - **same sheet** if `|o − p| < S/2`, roughly 2 … 16 vx;
  - `o ≈ −5 vx` is the vertical of the touching sheet, **not** the same sheet;
  - in general `round((o − p) / S)` = how many layers apart the two fibres are.
- **Verified span** of a horizontal = the x-range between its first and last same-sheet crossing with no
  wrong crossing in between. Only verified parts should be trusted.

---

## 3. The three scripts

### 3.1 `analyze_fiber.py`: trace and analyse a fibre from a point

```powershell
python analyze_fiber.py <x> <y> <z> --fibers <fibre folder> --segment $SHEET [--family auto|h|v]
```

What it does:

1. **Picks the family.** It compares the horizontal and vertical probabilities within `--seed-radius`
   (default 4 vx) and predicts missing tiles if needed. Give `--family h` or `--family v` to force it.
2. **Traces the fibre.**
   - Horizontals use the **x-monotonic, z-locked tracer** (`--xmono`): x always advances, y follows the
     fibre (so it can follow the sheet's curvature), and z may move only 0.25 vx per 2-vx step.
   - Verticals use the free 3D tracer.
   - Use `--free` to trace a horizontal with the free tracer instead.
3. **Writes the fibre into `--fibers`.**
   - `<name>.json` is a `vc3d_fiber` v1 file; VC3D can import it.
   - `<name>.path.npy` has columns x, y, z, prob, arc.
4. **Cross-checks it.** If the folder holds fibres of the other family, it computes all crossings and
   refits the layer model. It prints every crossing with its verdict (same sheet / n layers ±normal),
   the fibre's layer and its verified span, and rebuilds `fibers_3d.html`.

Example: trace a horizontal into a copy of the included set and see where it agrees with the verticals:

```powershell
Copy-Item -Recurse data\grid1_final7 data\my_fibers
python analyze_fiber.py 4500 3420 13470 --fibers data\my_fibers --segment $SHEET
```

```
family: horizontal-prob 1.00, vertical-prob 0.64 within 4 vx -> horizontal
traced 2562 vx (23.99 mm) ...
crossings of h_4500_3420_13470 (offset = horizontal minus vertical along the sheet normal):
  v_3631_3498_13153      at [3673.5 3552.5 13536.6]    offset   +5.8 vx -> same sheet
  v_3735_3456_12690      at [3786.2 3558.1 13524.5]    offset   -9.8 vx -> 1 layer(s) -normal
  v_4393_3359_12670      at [4506.2 3418.9 13467.0]    offset  +13.9 vx -> same sheet
```

The more crossings it has with **same sheet**, the more you can trust the fibre. If you need more
verticals somewhere, `tools/find_vertical_seeds.py` lists same-sheet vertical seeds along any horizontal.

### 3.2 `extend_fiber.py`: extend a horizontal monotonically

```powershell
python extend_fiber.py <fibre folder>\<name>.path.npy --segment $SHEET [--direction +x|-x|both]
```

Free tracing degrades toward the ends of a fibre: it wobbles up and down, then hops layers. This script:

1. **Winds back.** It finds the first stretch past the seed where the path wobbles sideways
   (RMS > max(1.2 vx, 2.5 × the fibre's own median)) or the mean probability drops below 0.6, and backs
   off `--backoff` vx (default 50) before it. To pick the restart point yourself, give `--from-x <x>`.
2. **Re-traces onward** one way with the x-monotonic, z-locked tracer:
   - x only advances in the chosen direction;
   - y re-centres up to `--y-slack` (default 2 vx) per step, following a smoothed slope;
   - z re-centres up to `--z-slack` (default 0.25 vx) per step;
   - faint gaps are bridged by looking up to `--reacquire-ahead` (default 60 vx) ahead.
3. **Splices and writes.** It joins the kept part and the extension and writes `<name>.json` /
   `<name>.path.npy`. The original is kept as `<name>.path.npy.orig` / `<name>.json.orig`; use `--out`
   to write into another folder instead.
4. **Reports crossings.** It prints every crossing of the result in x order, so you can see where the
   extension leaves the sheet. Cut it there with `--stop-x` or with `tools/fiber_trim.py`.

Example:

```powershell
python extend_fiber.py data\my_fibers\h_3417_3513_13514.path.npy --segment $SHEET
```

```
  +x: restart at x 4757 (was clean), extension reaches x 6105
crossings (same sheet = offset within 7 vx of +9):
  x   4502  v_4393_3359_12670      offset   +5.2 -> same sheet
  x   5152  v_5002_2937_12640      offset   -5.3 -> 1 layer(s) -normal   <- extension hops here
```

Then keep only the part that is still on the sheet:

```powershell
python tools\fiber_trim.py data\my_fibers data\my_fibers --cut h_3417_3513_13514:5000
```

### 3.3 `fibers_to_tifxyz.py`: build the sheet from the fibres

```powershell
python fibers_to_tifxyz.py <fibre folder> $SHEET <paths folder>\<new segment name> [options]
```

How it works:

1. **Samples the fibres.** Every fibre is sampled every `--step` vx (default 15). It skips `--trim` vx at
   each end and samples with probability below 0.5.
2. **Moves each sample to the sheet's mid-plane.** Horizontals move −p/2 and verticals +p/2 along the
   normal, so the surface runs between the two plies. `p` comes from `layers.json`.
3. **Projects onto the reference sheet.** Each sample gets its grid position (u, v) and its signed
   distance from the reference sheet along the normal.
4. **Fits a smoothing thin-plate spline** of that distance over (u, v). `--smooth` (default 50) controls
   it: larger is stiffer.
5. **Moves the grid.** Every grid vertex moves along its normal by the spline value. Vertices outside the
   area the fibres span (convex hull + `--margin` cells, default 5 = 100 vx) become −1 (invalid).
6. **Writes the segment.** `x.tif`, `y.tif`, `z.tif` (float32), `generations.tif`, and `meta.json` with
   uuid = folder name, bbox and area recomputed, and the same grid size and scale as the reference. Renders
   of the old and new sheet therefore line up pixel for pixel.

Example: a segment corrected from the included fibres:

```powershell
$PATHS = "C:\Users\ChenJeff\.VC3D\remote_cache\open_data\projects\paths"
python fibers_to_tifxyz.py data\grid1_final7 $SHEET "$PATHS\fiber_tps_example"
```

```
3553 fibre samples from 24 fibres; offsets to the reference sheet: median -2.1, 5-95% -15.0..+12.0 vx
thin-plate spline fitted in 0.8 s; residual at fibre samples: median |r| 1.69, 95% 5.53, max 32.72 vx
```

Read the per-fibre residuals at the end of the output: a fibre with a large residual disagrees with its
neighbours. Leave it out with `--exclude name1,name2`, or restrict it with
`--keep name:axis:lo:hi`.

---

## 4. Loading results in VC3D

- **Fibres:** Fibers panel → **Import** → pick `<name>.json`. Files are `vc3d_fiber` version 1; VC3D
  exports them as version 4.
- **Surface:** the output of `fibers_to_tifxyz.py` is a normal tifxyz segment. Put the folder in your
  project's `paths/` directory and reload the surfaces. It shows up under its folder name.
- `fibers_3d.html` in a fibre folder is a stand-alone interactive 3D view (plotly):
  - the sheet in grey;
  - fibres coloured by layer (dotted = horizontal, solid = vertical);
  - crossings in green (same sheet) or red (not).

---

## 5. How the sheet was made (full process)

The surface is not drawn from scratch. It is fitted to traced fibres, flattened, corrected by hand where
it still cuts a bend, and the fibres are then re-cleaned against the corrected sheet. All steps below use
scripts in `tools/` (the paths in many of them point at the `~/.VC3D` layout of the machine they were written on;
the constants are at the top of each script).

```
 seeds ─► trace (xmono / free / Viterbi march) ─► crossings ─► trim to verified span ─► fibre set (grid1_final*)
                                                                                          │
   pseudo supports (make_pseudo.py, edit areas only) ─────────────────────────────────────┤
                                                                                          ▼
                              fiber_surface_tps.py  ─► paths/fiber_only_<date><letter>  (TPS surface)
                                                                                          ▼
                              tools/flatten/slim_flatten.ps1 ─► ..._flatboi  (SLIM-flattened tifxyz)
                                                                                          ▼
                              hand correction in VC3D ─► ..._corrected / ..._PRIZESHEET
                                                                                          ▼
   surface-first cleaning: surf_trace, fiber_check, lat_check, touch_verticals, path_scan ─► data/cleaned/fibers.json
```

### 5.1 Trace the fibres

1. **Seed and trace** with `analyze_fiber.py` / `tools/trace_fiber_3d.py`. Horizontals: `--xmono` (x
   monotonic, y free with a slope prior, z nearly locked) is far better than the free 3D tracer; matched free
   traces stayed within 1.5 vx in clean parts and reached x ≈ 6600. Locking the tracer to the reference sheet's
   normal (`--lock-layer`) hurt: where the sheet is off-layer the normal is wrong and the probability drops.
2. **Refine and extend** with the Viterbi tools instead of the plain tracer:
   - `fiber_march.py` (Viterbi in the fibre's own frame; horizontals: depth = xy-perpendicular, side = z;
     verticals use `--ref <surface>`), batch-run with `march_batch.ps1`;
   - `fiber_dpcore.py` is the shared DP. `lock_strand` (straight guide σ = 150 vx, side ±24 vx, sideways cost 5)
     follows **one** physical strand and removes the shake;
   - `fiber_straight.py` reports lateral shake and can lock it;
   - `fiber_anchor_trace.py` is the best tool for *new* fibres: it anchors the path one ply away from every
     crossing fibre of the other family and picks the strand with a Viterbi pass. It gave 9/9 … 12/12 consistent
     fibres where free seeds landed one layer off. `fiber_seed2.py` seeds the same way (use `--ply -9`; `+9`
     lands one layer off). `eval_new.ps1` checks a candidate's crossings quickly;
   - hard stretches (bends, damaged zones): `fiber_slice_zip.py` (fixed-z slice zipper, `--ztrack`, left pin;
     reproduced three reference horizontals to within 1–5 vx through bends), `fiber_rebuild_region.py`,
     `fiber_zip.py`, `fiber_zmarch.py`, `retrack_h.ps1` (re-trace a horizontal pinned at the verticals' crossings).
     `fiber_streamline.py` was not reliable.
3. **Verticals.** Real verticals lean dx/dz ≈ +0.1 … +0.29. Verticals seeded at a fixed x (`fiber_anchor_trace
   --at X`) come out with dx/dz ≈ 0, cut across strands and are shaky: do not use them. `canon_trace.py` follows
   the tangent field of the trusted verticals (surface − 9·normal, anchor prior to the horizontals) and gives clean
   0.2–1.5 vx shake; `vert_audit.py` reports each vertical's direction against that canon.

### 5.2 Verify and trim

- **Crossings.** `fiber_crossings_fn.py` (normal from the two fibres, no surface) writes `crossings.csv`;
  `fiber_xline.py` prints a per-fibre crossing string (window p = 9, S = 14); `fiber_layers.py` refits S, p and an
  integer layer per fibre (`layers.json`; re-run it after adding fibres, otherwise the viewer shows layer −1).
- **Trim** each fibre to the span its crossings confirm: `fiber_trim_verified.py` (margin 200 vx, ≥ 3 good
  crossings), `trim_iterate.ps1` / `trim_loop.ps1` (alternate horizontals and verticals until nothing changes).
  `fiber_fix_excursions.py` repairs an isolated bad crossing with a pinned self-bridge instead of cutting.
  `fiber_dupes.py` / `fiber_merge.py` find and join duplicates. `fiber_hops.py` finds layer changes.
- **Marching extensions** often hop at bends and weak zones, and near the scroll edges they climb sheets: always
  crop to the crossing-verified span and check `fiber_offsets.py` against a trusted sheet.
- **Traps.**
  - Consistent crossings do **not** prove the right sheet. In a bend the whole grid can consistently cut a corner
    (all fibres hop together). Compare against a sheet that follows the bends.
  - Crossing and surface checks cannot see an in-sheet lateral hop. Compare the z-gap to neighbouring fibres
    (a step of more than ~15 vx at a non-bend is a hop) and use `lat_check.py`.
  - Damaged zones show layer dislocations; trust only fibres consistent at all crossings. Region 2
    (x 3450–3750, z 11500–11950) has a lens/dislocation where left-continuous traces land 1–2 layers off.
  - Near the bend at (3707, 3410, 11814) crossing offsets use smooth-TPS normals and are unreliable.
- Rules of thumb used throughout: horizontals sit at normal offset 0 and verticals at −9 (the ply offset)
  relative to the sheet; a crossing is good when `|offset − 9| < 6` and clearly bad when `|offset − 9| ≥ 9`.

### 5.3 The fibre set

`data/grid1_final26` is the best set from this process (38 horizontals, 78 verticals, crossings 97 % good,
x 2199–6581, z 10776–13722). Its history, in short: `grid1_final7` (verified set, section 6) → `final19` (three
bad verticals removed, region-A verticals bridged) → `final20` (horizontals snapped to the reference sheet in region A and
the bumps at (2885, 12719), (5264, 12640), (5892, 13241)) → `final21` (low-z snaps) → `final23` (all low-z horizontals
re-anchored to the verticals with `retrack_h.ps1` / `brute_levels_ref.ps1`) → `final24` (excursion fixes) →
`final25` (`fiber_trim_verified.py`, margin 200, min 3 good) → `final26` (duplicates removed; verticals joined:
`k4610 + j4650` → `v_k4610_12900`, `k4400 + k4360` → `v_k4400_11800`).

### 5.4 Fibre-only surface (TPS)

```powershell
python tools\fiber_surface_tps.py data\grid1_final26 "$PATHS\fiber_only_<date><letter>" --plane h --hweight 10 --hstep 8 --no-hull --reach 1000
```

- `--plane h` puts the surface on the horizontals' face (verticals are moved a full ply onto it); `--hweight 10 --hstep 8`
  makes it follow the horizontals closely (residual < 1.5 vx). These flags reproduce the earlier `fiber_only_*`
  surfaces (vs `final18b`: dy 5–95 % −1.9 … +2.4); `--reach 1000` matches the old bounding box (reach 400 gives 13 cm²
  instead of 22). It takes minutes.
- The TPS extrapolates badly where no fibres exist (e.g. +100 vx at x 3800–4200, z 10800–11300, or −50 … −60 at
  z 13400–13800). Fix: `make_pseudo.py` creates pseudo horizontals from the hand-fixed sheet only where no fibre
  lies within ~150 vx. They are used for the TPS build only and are never exported as fibres.
- Compare a fibre set with a hand-fixed sheet objectively: build the TPS (`tps_eval.ps1`) and run
  `dy_map.py <hand sheet> <new TPS>` (table of dy over x, z). `surf_resample.py` resamples a hand-edited tifxyz onto
  the TPS grid first.
- `meta.json`: `uuid` **and** `seed_surface_id` must equal the folder name (human readable). The flatten step below
  rewrites both; check this after any tool that writes a surface.

### 5.5 Flatten

```powershell
powershell -ExecutionPolicy Bypass -File tools\flatten\slim_flatten.ps1 -Segment "$PATHS\fiber_only_<date><letter>"
```

`vc_tifxyz2obj` → `tools/flatten/flatboi.py` (libigl SLIM, a Windows stand-in for VC3D's flatboi) → `vc_obj2tifxyz`.
The result is `<surface>_flatboi`. The paths at the top of `slim_flatten.ps1` must point at your VC3D `bin` and
Python environments. In the flattened grid, columns follow the verticals and rows follow the horizontals (horizontals
are nearly straight lines in (row, col)), which the surface-first tools rely on. VC3D's abf++ flatten rewrites a
surface **in place**: keep a copy of any hand-corrected sheet (VC3D autosaves to `paths/backups/<name>/0..9`).

### 5.6 Hand correction, then surface-first cleaning

The flattened surface was corrected by hand in VC3D at the bends (`..._flatboi_corrected`, later `..._PRIZESHEET`).
From there the process runs the other way round: the **sheet leads and fibres stick to it**. Only fibres that pass
the checks below are kept in the final collection.

- **Tracing along the sheet.** `surf_trace.py` follows a strand along the sheet's own (row, col) grid from a seed (a
  seed clicked on the sheet is at offset ≈ 0, so a vertical is started on the −9 plane), blending the guide between
  the two bracketing fibres, with `lock_strand` (`--ldepth 2` sticks tighter). `touch_verticals.py` re-traces each
  vertical from its crossing with a given horizontal so it reaches and touches it.
- **Finding a strand at a target z.** `build_hz_grid.py` samples the horizontal probability on the sheet,
  `path_scan.py` finds the best distinct free Viterbi strands through it (a sideways step costs λ, so a path follows one
  strand), `refine_path.py` turns one into a fibre, `guide_scan.py` / `line_search.py` score blended or straight guides.
- **Confidence.** `fiber_check.py` bins a fibre (150 vx) and takes the **median** probability per bin (horizontal
  probability dips at crossings), its deviation from the sheet and its crossings with the trusted fibres. A bin is
  confident if p ≥ 0.5 (or a good crossing and p ≥ 0.05), |dev| ≤ 6 vx and no clearly bad crossing; `--write` keeps the
  longest confident run. `lat_check.py` checks in-sheet wander against a smoothed copy. Neighbour z-gaps must change
  smoothly (see traps above).
- **Collections.** `filtered_io.py export|update|remove` converts VC3D fibre collections (`filtered.json`) to and from
  `<name>.path.npy` files; every write makes a timestamped backup.
- The result of this stage is `data/cleaned/fibers.json` (section 6).

---

## 6. Included data

### `data/cleaned/fibers.json`: the latest cleaned fibres

A VC3D fibre collection (`vc3d_fiber_collection`, version 1) with the 15 hand-cleaned fibres that stick to the
corrected sheet: horizontals `h_3297_3319_12216`, `h_3313_3375_12532`, `h_3326_3423_12750`, `h_3369_3475_13176`,
`h_3394_3491_13385`, `h_4759_2880_11817` and verticals `v_3223_3415_12778`, `v_3769_3518_13203`, `v_3980_3486_12819`,
`v_4366_3355_12573`, `v_4659_3250_12716`, `v_4974_2912_12509`, `v_5325_2599_12538`, `v_5856_2645_12464`,
`v_6138_2808_12396`. Import it in the Fibers panel, or convert it to `.path.npy` with
`python tools\filtered_io.py export data\cleaned\fibers.json <out dir>`.

### `data/grid1_final26/`: the full fibre set

The best full set (38 horizontals, 78 verticals, each as `.json` and `.path.npy`, plus `crossings.csv` and
`layers.json`; process in section 5.3). It is the input of the TPS surface `fiber_only_20261102b`.

### `data/grid1_final7/`

The verified fibre set behind the fibre-only surface `fiber_only_20261029d`, traced on PHerc0846A around
x 1700–6600, z 10700–13800:

- 32 horizontals and 30 verticals, each as `.json` (VC3D, with the coordinate binding to volume
  20250728152254 at full resolution, so VC3D doesn't ask for units) and `.path.npy`.
- Analysis outputs: `crossings.csv`, `layers.json`, `fibers_3d.html` and `fibers_overview.png`.
- Every fibre is cut to the stretch its crossings confirm; `tools/fiber_hops.py` finds one layer change in
  the whole set (on `v_5982`, far right).

The surface was built without any reference sheet:

```powershell
python tools\fiber_surface_tps.py data\grid1_final7 "$PATHS\fiber_only_20261029d" --verified-margin 200 --reach 1000 --no-hull --smooth 15
```

How two hard regions were resolved:

- **V-bend near (3707, 3410, 11814).** Below z ≈ 12000 the left verticals (`v_2905`, `v_3631`, `v_3735`,
  `v_3774`) are on the neighbouring layer; they are cut at z 12060. The lower region follows the layer of
  `v_4120` (unbroken band from z 10700 to 13535), anchored by `v_4355`, `v_3039` and `v_3242`.
- **Bend at x 5000–5650, z 12775–13300.** Horizontals hop inside the bend, so pieces stop short of it on
  each side; `v_5577` (unbroken band) and `v_5372` anchor the right side, `h_3394` and `h_5508` cross it.

Known caveats:

- The left half at z 11800–12060 and x 4800–5000 below z 11900 have few fibres; the surface interpolates there.
- Beyond x ≈ 6200 and at the extrapolated edges the surface is pure extrapolation.

---

## 7. Helper tools (`tools/`)

The surface-first and tracing scripts of section 5 are all in this folder (`tools/*.py`, `tools/*.ps1`); the table lists the
older helpers. Every script prints its options with `-h` (the `.ps1` files document their parameters in a header comment).
The `extend_bridge.py` here is the bridging/hopping variant of `extend_fiber.py` (usage in its docstring).

| Script | Purpose |
|---|---|
| `trace_fiber_3d.py` | The tracer: `--xmono` / `--zmono` (monotonic in x / z), `--slope-rate` (sharper bends), `--one-way`, `--axis`, `--seed-radius`, … (`-h` for all) |
| `extend_mono.py` | Extend a fibre past one end with the monotonic tracer (keeps `.orig` backups) |
| `fiber_surface_tps.py` | Fibre-only surface: thin-plate spline y = f(x, z) on the fibres' mid-plane, no reference sheet (`--no-hull`, `--reach`, `--smooth`) |
| `fiber_crossings.py` | Closest approach and signed normal offset for every vertical × horizontal pair → `crossings.csv` |
| `fiber_crossings_fn.py` | Same, but the normal comes from the two fibres at each crossing (no surface needed) |
| `fiber_trim_crossings.py` | Cut every fibre to the stretch its crossings confirm (blame-based) |
| `fiber_hops.py` | Hop detector: fewest layer changes along fibres that explain all crossings; `--cut` keeps the longest consistent stretch |
| `fiber_lsolve.py` | Integer layer assignment over fibre pieces |
| `band_trim.py`, `band_check3d.py`, `hop_screen.py` | Cut / screen fibres where their own probability band breaks or changes |
| `fiber_side_audit.py`, `fiber_order_check.py` | Per-bin side check of fibres against a surface; parallel fibres must not cross |
| `slab_view.py`, `slab_z.py`, `fiber_section.py` | Images: fixed-x / fixed-z slabs and sections along a fibre (CT + probabilities) |
| `fiber_meta.py` | Coordinate binding written into fibre JSON; run on a folder to add it to existing files |
| `fiber_layers.py` | Fits spacing S, ply offset p and an integer layer per fibre → `layers.json` (use `--min-spacing 14 --max-spacing 20`) |
| `fiber_coverage.py` | Verified same-sheet x-range of every horizontal |
| `find_vertical_seeds.py` | Same-sheet vertical seed points along a horizontal (`h_name 5300:6400:6` scans x) |
| `fiber_trim.py` | Copy a fibre set, cutting fibres to ranges (`--cut name:x`, `--keep name:axis:lo:hi`) and dropping fibres (`--drop`) |
| `fiber_view3d.py` | Rebuilds `fibers_3d.html` and `fibers_overview.png` |
| `fiber_maps.py` | Horizontal / vertical fibre-depth CT maps in the sheet's (u, v) frame |
| `fiber_ribbon.py` | Per-fibre "ribbon": CT at depths −15 … +15 vx along the fibre |
| `fiber_hcheck.py` | Each horizontal's signed offset from the reference sheet per x bin |
| `afv_core.py` | CT access, prediction store, on-demand tile prediction (used by the others) |

Every script prints its options with `-h`.

---

## 8. Troubleshooting

- **`no h fibre within 4 vx of the point`:** the point is not on a fibre of that family. Raise
  `--seed-radius` (up to ~12) or pick the point in the model's prediction (the message also reports the
  other family's probability).
- **Very slow first run in a new area:** the model is predicting tiles and CT chunks are downloading.
  Later runs in the same area read the caches.
- **PowerShell:** inline `python -c` code containing braces breaks; use script files. If a `.ps1` prints
  nothing when called with `&`, run it with `powershell -ExecutionPolicy Bypass -File <script.ps1>`.
  Variable names are case-insensitive (a loop variable `$s` overwrites a parameter `$S`), a bare `--` is stripped
  from native command arguments (the brute-force scripts use `::` instead), and unquoted `Key=a,Value=b`
  arguments become arrays.
- **Windows paths are case-insensitive:** `grid1_R28` and `grid1_r28` are the same folder; copying one onto the other
  loses files (`trace_candidates.ps1` must never use `-Src` equal to `-OutDir`).
- **FIPS mode:** MD5 is blocked; use SHA256 when hashing files.

---

## 9. Credits and licences

- **Fibre model:** AFV fibre model V2 by **Qualzz20**, licensed under Apache-2.0, building on the
  [AFV ~9 µm fibre model](https://huggingface.co/Qualzz20/afv_fiber_9um). All fibre probabilities and
  directions used here come from this model. The model code (`model.py`) and checkpoint are not
  redistributed in this repository; they are loaded from `AFV_MODEL_DIR`.
- **CT data:** PHerc0846A, volume 20250728152254, from the
  [Vesuvius Challenge](https://scrollprize.org) open data. Use of the data is subject to the Vesuvius
  Challenge data terms.
- **VC3D / tifxyz:** the surface and fibre formats follow VC3D (Volume Cartographer 3D).
- **This repository:** no licence has been chosen yet, so all rights are reserved by default.
