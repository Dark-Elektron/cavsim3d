# cavsim3d test bench (ngapp)

CST-style studio for exercising the cavsim3d pipeline interactively:
geometry modelling with primitives (waveguide / cylinder / box / sphere) and
project import, on-the-fly face naming + materials, meshing, the staged
FOM → ROM → concatenation pipeline, S/Z sweeps vs. the analytical solution,
and 3D fields reconstructed per section **from the coupled ROM**.

## Run

From the cavsim3d repo root (conda env `cavsim3d`):

```bash
pip install -e testbench/     # once
python -m testbench           # serve (opens the browser)
python -m testbench --dev     # hot reload while editing the app
```

**Connect with the FULL URL the server prints** — it carries the websocket
port and token (`http://localhost:8765?backendPort=…&wsToken=…`). A plain
`http://localhost:8765` loads an empty shell and stays blank.

Every run writes a real, inspectable project to `<repo>/testbench_runs/<name>`.

## Layout

```
testbench/
├─ pyproject.toml
├─ README.md
├─ .github/workflows/deploy.yml     (optional platform deployment)
└─ src/testbench/
   ├─ __init__.py
   ├─ __main__.py                   (python -m testbench)
   ├─ app.py                        (the UI / pipeline driver)
   └─ appconfig.py                  (ngapp AppConfig entry point)
```
