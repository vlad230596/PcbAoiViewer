# PcbAoiViewer — notes for agents

Project goal, stage status, results, rationale and roadmap live in `README.md` (Russian). Read it before changing the stitcher or picking the next stage.

## Working here

- Python: `venv/Scripts/python.exe` (Windows, Python 3.12, OpenCV 4.9, numpy). scipy is not installed; numpy least squares is enough so far.
- Current pipeline: `GridStitcher.py` (algorithm), `stitchGrid.py` (CLI), `BuildViewer.py` (Deep Zoom tiles + `index.html`, stack viewer + `3d.html`). Multi-height: `FocusStack.py` (algorithm), `stackFocus.py` (CLI) — reference height stitched by `GridStitcher`, every other height mapped onto it by one similarity, depth from focus, ortho-corrected second pass, all-in-focus. Old pipeline (`main.py`, `MainStitcher.py`, `PlaneStitcher.py`, `ViewDrawer.py`, `Dataset.py`, `test.py`) is history; leave it untouched.
- Outputs go to `out/` (git-ignored). Datasets (`SocketTester2/`, `captured*`) are git-ignored and large; read only.
- Code style: camelCase methods, small classes, thresholds given in full-resolution px and scaled by `self.scale` inside `GridStitcher`.
- Quick iteration: `--scale 0.5 --no-viewer --seam-crops 0` runs in ~15 s; full resolution ~30 s.
- Stack: `stackFocus.py "SocketTester2/captured_*" --ref 51.5 --scale 0.5 --no-viewer` ~4 min; `--ortho-passes 0` halves it when only the depth matters.

## Verifying a change

Numbers alone miss seam artefacts. After a stitching change, crop the panorama at full resolution on seams over text and IC pins, on both tantalum capacitors and the 220-16 capacitor pair, and on the red connectors, then look at the crops. Compare against the previous output folder. `report.json` holds residual stats and timings.

For the focus stack, crop the same places in `focus.jpg` (all-in-focus) and `layers/z*.jpg`, look at `height_color.png`, and open `3d.html` in a browser. For the marking layer, compare IC tops in `marking.jpg` against `focus.jpg` (the QFN under frame 50_60 is the hard case) and check `marking.json` for the chosen frames.

## Gotchas

- Off-board background (the mat) lies at another depth: its pairs disagree by 50–175 px and are rejected on purpose (`--max-deviation`). Parallax regions there are expected.
- Z is camera height (focus plane moves up with it); camera ~89.6 mm above the board. `captured_45.0..49.5` were shot at another XY offset and focus below the board; the stack uses Z ≥ 50.
- The camera is rotated ~47° to the machine axes, so diagonal grid neighbours overlap strongly and are real pairs.
- In legacy `Dataset.at(row, col)` "row" is machine X and Y is inverted.
- `capture.py`: `CncController.home()` passes an extra `serial` argument to `command()` and will fail.
