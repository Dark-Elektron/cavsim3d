# -*- coding: utf-8 -*-
"""cavsim3d test bench (ngapp) — CST-style studio.

    +----------------------------------------------------------------------+
    | title   [Geometry | Meshing | Boundary cond. | Simulation | Results] |
    | ribbon of the active mode (primitives / mesh / BC / solve / field)   |
    +--------------+-------------------------------------------------------+
    | project tree |  viewport tabs: Geometry | Mesh | 3D field | S | Z    |
    | (components, |  (webgui / plotly fill the region; drag the splitter, |
    |  results)    |   fullscreen button on the 3D views)                  |
    +--------------+-------------------------------------------------------+
    | status line + expandable Messages log (full errors readable)         |
    +----------------------------------------------------------------------+

Geometry modelling goes through cavsim3d primitives (RectangularWaveguide,
CircularWaveguide=cylinder, Box, Sphere) and ``proj.fds.import_model`` — each
solid is a tree entry (Components) with name/material/face names set on the
fly in its dialog.  Runs use the standard pipeline; the 3D field is
reconstructed per section FROM THE COUPLED ROM.  Solids with repeat counts or
imports run the netlist path; multiple raw solids can instead be GLUED into
one conformal mesh (Meshing ribbon); a single solid runs fds.fom.rom.
"""

from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path

import numpy as np

from ngapp.app import App
from ngapp.components import (
    Col,
    Div,
    Heading,
    Label,
    NumberInput,
    PlotlyComponent,
    QBtn,
    QCard,
    QCardSection,
    QDialog,
    QIcon,
    QInput,
    QItem,
    QItemSection,
    QList,
    QMenu,
    QSelect,
    QSeparator,
    QSlider,
    QSpace,
    QSplitter,
    QTab,
    QTabPanel,
    QTabPanels,
    QTabs,
    QToggle,
    QToolbar,
    QTree,
    Row,
    WebguiComponent,
)

# <repo>/testbench_runs (src/testbench/app.py -> up 3 = repo root)
RUNS_DIR = Path(__file__).resolve().parents[3] / "testbench_runs"

TAB_GEO, TAB_MESH, TAB_FIELD, TAB_S, TAB_Z = "geo", "mesh", "field", "s", "z"
MODE_GEO, MODE_MESH, MODE_BC, MODE_SIM, MODE_RES = (
    "m_geo", "m_mesh", "m_bc", "m_sim", "m_res")
# which viewport tab each ribbon mode focuses
_MODE_FOCUS = {MODE_GEO: TAB_GEO, MODE_MESH: TAB_MESH, MODE_BC: TAB_GEO,
               MODE_RES: TAB_S}

_FILL = "width: 100%; height: 100%; min-height: 0"


def _num(label, val, step="any", width=110):
    return NumberInput(ui_label=label, ui_model_value=val, ui_step=step,
                       ui_dense=True, ui_outlined=True,
                       ui_style=f"width: {width}px", ui_class="q-mx-xs")


def _txt(label, val, width=140):
    return QInput(ui_label=label, ui_model_value=val, ui_dense=True,
                  ui_outlined=True, ui_style=f"width: {width}px",
                  ui_class="q-mx-xs")


def _fill_webgui(comp: WebguiComponent) -> WebguiComponent:
    """Make a webgui view fill its container (ctor hard-codes 500px) and give
    it fullscreen / reset-view canvas buttons."""
    comp.ui_style = _FILL
    comp.slot_buttons = [
        WebguiComponent.canvas_button(
            ui_icon="mdi-fullscreen", ui_tooltip="Fullscreen",
            on_click=lambda _: comp.toggle_fullscreen()),
        WebguiComponent.canvas_button(
            ui_icon="mdi-eye-refresh-outline", ui_tooltip="Reset view",
            on_click=lambda _: comp.set_camera()),
    ]
    return comp


class TestBench(App):
    def __init__(self, **kwargs):
        # -------- model state ------------------------------------------
        self._solids: list[dict] = []   # {name, kind, geo|path, n, params}
        self._selected: str | None = None
        self._proj = None
        self._concat = None             # coupled system (netlist/glued runs)
        self._freqs = None
        self._S = self._Z = self._Sa = None
        self._fig_s = self._fig_z = None
        self._busy = False
        self._messages: list[str] = []

        self._build_dialogs()
        self._build_ribbons()
        self._build_tree()
        self._build_viewport()
        self._build_messages()

        body = QSplitter(ui_model_value=18, ui_unit="%", ui_limits=[10, 45],
                         ui_style=_FILL)
        body.ui_slot_before = [self._tree_panel()]
        body.ui_slot_after = [self._viewport]

        # Horizontal splitter: drag the handle to pull the Messages panel up
        # or down.  Sized from the bottom (reverse) in pixels.
        main = QSplitter(ui_horizontal=True, ui_reverse=True, ui_unit="px",
                         ui_model_value=64, ui_limits=[28, 600],
                         ui_style="flex: 1 1 auto; min-height: 0")
        main.ui_slot_before = [Div(body, ui_style=_FILL)]
        main.ui_slot_after = [self._bottom]

        super().__init__(
            Div(
                self._header,
                self._ribbons,
                main,
                *self._dialogs,
                ui_class="column no-wrap",
                ui_style="height: 100vh",
            ),
            **kwargs,
        )
        self._log("Ready. Add solids in the Geometry ribbon, then Run "
                  "(Simulation ribbon).")

    # ================================================================== #
    # dialogs (floating parameter ribbons for each primitive)
    # ================================================================== #
    def _build_dialogs(self):
        self._dialogs = []

        def dialog(title, fields, on_add):
            btn = QBtn(ui_label="Add", ui_color="primary", ui_class="q-ma-sm")
            btn.on("click", on_add)
            dlg = QDialog(QCard(
                QCardSection(Heading(title, 6)),
                QCardSection(Row(*fields, ui_class="wrap items-center")),
                QCardSection(btn),
                ui_style="min-width: 540px",
            ))
            self._dialogs.append(dlg)
            return dlg

        # rectangular waveguide -----------------------------------------
        self.rw = dict(name=_txt("name", "wg1"), a=_num("a [m]", 0.1),
                       b=_num("b [m]", 0.05), L=_num("L [m]", 0.06667),
                       maxh=_num("maxh [m]", 0.06), n=_num("repeat n", 1, "1"),
                       mat=_txt("material", "vacuum"))
        self.dlg_rw = dialog("Rectangular waveguide", list(self.rw.values()),
                             lambda: self._add_solid("rwg"))

        # circular waveguide (cylinder) ----------------------------------
        self.cy = dict(name=_txt("name", "cyl1"), r=_num("radius [m]", 0.05),
                       L=_num("length [m]", 0.06667),
                       maxh=_num("maxh [m]", 0.03), n=_num("repeat n", 1, "1"),
                       mat=_txt("material", "vacuum"))
        self.dlg_cy = dialog("Cylinder (circular waveguide)",
                             list(self.cy.values()),
                             lambda: self._add_solid("cyl"))

        # box (face names editable on the fly) ---------------------------
        self.bx = dict(name=_txt("name", "box1"), a=_num("x [m]", 0.1),
                       b=_num("y [m]", 0.05), L=_num("z [m]", 0.06667),
                       maxh=_num("maxh [m]", 0.06), n=_num("repeat n", 1, "1"),
                       mat=_txt("material", "vacuum"),
                       f_zmin=_txt("face z-min", "port1", 100),
                       f_zmax=_txt("face z-max", "port2", 100),
                       f_ymin=_txt("face y-min", "bottom", 100),
                       f_ymax=_txt("face y-max", "top", 100),
                       f_xmin=_txt("face x-min", "left", 100),
                       f_xmax=_txt("face x-max", "right", 100))
        self.dlg_bx = dialog("Box (name faces on the fly)",
                             list(self.bx.values()),
                             lambda: self._add_solid("box"))

        # sphere ----------------------------------------------------------
        self.sp = dict(name=_txt("name", "sph1"), r=_num("radius [m]", 0.05),
                       maxh=_num("maxh [m]", 0.03),
                       mat=_txt("material", "vacuum"))
        self.dlg_sp = dialog("Sphere (closed resonator, no ports)",
                             list(self.sp.values()),
                             lambda: self._add_solid("sphere"))

        # import from an already-run project ------------------------------
        self.im = dict(name=_txt("name", "imported"),
                       path=_txt("project path", str(RUNS_DIR / "bench"), 320),
                       n=_num("repeat n", 1, "1"))
        self.dlg_im = dialog("Import model from project",
                             list(self.im.values()),
                             lambda: self._add_solid("import"))

    # ================================================================== #
    # ribbons: Geometry | Meshing | Boundary conditions | Simulation | Results
    # ================================================================== #
    def _build_ribbons(self):
        def rbtn(label, icon, fn, color="primary", flat=True):
            b = QBtn(ui_label=label, ui_icon=icon, ui_color=color,
                     ui_flat=flat, ui_dense=True, ui_no_caps=True,
                     ui_class="q-mx-xs")
            b.on("click", fn)
            return b

        def ribbon(*children):
            # single compact strip; keep vertical footprint minimal
            return Row(*children,
                       ui_class="items-center no-wrap q-gutter-x-xs q-px-sm",
                       ui_style="min-height: 44px; overflow-x: auto")

        # --- Geometry ----------------------------------------------------
        geo_ribbon = ribbon(
            rbtn("Waveguide", "crop_16_9",
                 lambda: setattr(self.dlg_rw, "ui_model_value", True)),
            rbtn("Cylinder", "circle",
                 lambda: setattr(self.dlg_cy, "ui_model_value", True)),
            rbtn("Box", "check_box_outline_blank",
                 lambda: setattr(self.dlg_bx, "ui_model_value", True)),
            rbtn("Sphere", "radio_button_unchecked",
                 lambda: setattr(self.dlg_sp, "ui_model_value", True)),
            QSeparator(ui_vertical=True),
            rbtn("Import project", "input",
                 lambda: setattr(self.dlg_im, "ui_model_value", True)),
            QSeparator(ui_vertical=True),
            rbtn("Delete selected", "delete", self._delete_selected,
                 color="negative"),
            rbtn("Clear all", "delete_sweep", self._clear_solids,
                 color="negative"),
        )

        # --- Meshing -------------------------------------------------------
        self.in_glue_maxh = _num("glued maxh [m]", 0.06)
        self.tgl_glue = QToggle(ui_label="glue into ONE conformal mesh "
                                         "(multi-solid)", ui_model_value=False)
        self.lbl_mesh = Label("", ui_class="q-ml-md text-caption")
        mesh_ribbon = ribbon(
            self.tgl_glue, self.in_glue_maxh,
            rbtn("Generate & view", "grid_on", self._on_generate_mesh),
            self.lbl_mesh,
        )

        # --- Boundary conditions -------------------------------------------
        self.bc_solid = QSelect(ui_label="solid", ui_options=[], ui_dense=True,
                                ui_outlined=True, ui_style="width: 130px")
        self.bc_face = QSelect(ui_label="face", ui_options=[], ui_dense=True,
                               ui_outlined=True, ui_style="width: 130px")
        self.bc_newname = _txt("new face name", "", 130)
        self.bc_mat = _txt("material", "vacuum", 120)
        self.bc_solid.on_update_model_value(lambda e: self._bc_refresh_faces())
        bc_ribbon = ribbon(
            self.bc_solid, self.bc_face, self.bc_newname,
            rbtn("Rename face", "drive_file_rename_outline", self._bc_rename),
            QSeparator(ui_vertical=True),
            self.bc_mat, rbtn("Set material", "texture", self._bc_material),
            Label("(faces named 'port*' become ports; renames re-mesh the "
                  "solid)", ui_class="text-caption q-ml-md"),
        )

        # --- Simulation ------------------------------------------------------
        self.in_name = _txt("project", "bench", 110)
        self.in_fmin = _num("fmin [GHz]", 1.8)
        self.in_fmax = _num("fmax [GHz]", 2.4)
        self.in_nsamp = _num("snapshots", 4, "1", 90)
        self.in_order = _num("order", 2, "1", 80)
        self.in_modes = _num("port modes", 1, "1", 90)
        self.in_tol = _num("ROM tol", 1e-9)
        self.in_sweep = _num("sweep pts", 201, "1", 90)
        self.tgl_async = QToggle(ui_label="background solve",
                                 ui_model_value=True)
        self.tgl_concat_rom = QToggle(ui_label="reduce concat",
                                      ui_model_value=False)
        self.btn_run = QBtn(ui_label="Run pipeline", ui_icon="play_arrow",
                            ui_color="primary", ui_dense=True,
                            ui_no_caps=True, ui_class="q-mx-sm")
        self.btn_run.on("click", self.on_run)
        sim_ribbon = ribbon(
            self.in_name, self.in_fmin, self.in_fmax, self.in_nsamp,
            self.in_order, self.in_modes, self.in_tol, self.in_sweep,
            self.tgl_concat_rom, self.tgl_async, self.btn_run,
        )

        # --- Results -----------------------------------------------------------
        self.sel_section = QSelect(ui_label="section", ui_options=[],
                                   ui_dense=True, ui_outlined=True,
                                   ui_style="width: 130px")
        self.sel_port = QSelect(ui_label="excite port", ui_options=[],
                                ui_dense=True, ui_outlined=True,
                                ui_style="width: 130px")
        self.sel_field = QSelect(ui_label="field", ui_model_value="E",
                                 ui_options=["E", "H"], ui_dense=True,
                                 ui_outlined=True, ui_style="width: 80px")
        self.sel_comp = QSelect(ui_label="component", ui_model_value="abs",
                                ui_options=["abs", "real", "imag"],
                                ui_dense=True, ui_outlined=True,
                                ui_style="width: 110px")
        self.sl_freq = QSlider(ui_min=0, ui_max=0, ui_step=1,
                               ui_model_value=0, ui_label=True,
                               ui_style="width: 220px", ui_class="q-mx-md")
        self.lbl_freq = Label("f = —", ui_class="text-caption")
        for c in (self.sel_section, self.sel_port, self.sel_field,
                  self.sel_comp):
            c.on_update_model_value(lambda e: self._draw_field())
        self.sl_freq.on_update_model_value(lambda e: self._draw_field())
        res_ribbon = ribbon(
            self.sel_section, self.sel_port, self.sel_field, self.sel_comp,
            self.sl_freq, self.lbl_freq,
            QSeparator(ui_vertical=True),
            rbtn("Export CSV", "download", self._export_csv),
            rbtn("Reveal folder", "folder_open", self._reveal_folder),
        )

        # --- header: title + mode tabs; ribbons switch with the mode --------
        self.mode_tabs = QTabs(
            QTab(ui_name=MODE_GEO, ui_label="Geometry"),
            QTab(ui_name=MODE_BC, ui_label="Boundary conditions"),
            QTab(ui_name=MODE_MESH, ui_label="Meshing"),
            QTab(ui_name=MODE_SIM, ui_label="Simulation"),
            QTab(ui_name=MODE_RES, ui_label="Results"),
            ui_model_value=MODE_GEO, ui_dense=True, ui_no_caps=True,
            ui_align="left", ui_class="text-primary",
        )
        self.mode_panels = QTabPanels(
            QTabPanel(geo_ribbon, ui_name=MODE_GEO, ui_class="q-pa-none"),
            QTabPanel(bc_ribbon, ui_name=MODE_BC, ui_class="q-pa-none"),
            QTabPanel(mesh_ribbon, ui_name=MODE_MESH, ui_class="q-pa-none"),
            QTabPanel(sim_ribbon, ui_name=MODE_SIM, ui_class="q-pa-none"),
            QTabPanel(res_ribbon, ui_name=MODE_RES, ui_class="q-pa-none"),
            ui_model_value=MODE_GEO,
        )
        self.mode_tabs.on_update_model_value(self._on_mode)
        self._header = QToolbar(
            QIcon(ui_name="cable", ui_size="sm"),
            Label("cavsim3d", ui_class="text-weight-bold q-mx-sm"),
            self.mode_tabs, QSpace(),
            ui_class="bg-grey-3 text-primary",
            ui_style="min-height: 36px",
        )
        self._ribbons = Div(self.mode_panels,
                            ui_class="bg-grey-1",
                            ui_style="border-bottom: 1px solid #ddd")

    def _on_mode(self, e):
        self.mode_panels.ui_model_value = e.value
        focus = _MODE_FOCUS.get(e.value)
        if focus:
            self._set_tab(focus)

    # ================================================================== #
    # tree + viewport
    # ================================================================== #
    def _build_tree(self):
        self.tree = QTree(ui_nodes=[], ui_node_key="id", ui_label_key="label",
                          ui_children_key="children",
                          ui_default_expand_all=True, ui_dense=True,
                          ui_selected="", ui_selected_color="primary")
        self.tree.on_update_selected(self._on_tree_select)
        self._refresh_tree()

    def _refresh_tree(self):
        solids = [{"id": f"solid:{s['name']}",
                   "label": s["name"] + (f"  (×{s['n']})" if s["n"] > 1 else ""),
                   "icon": {"rwg": "crop_16_9", "cyl": "circle",
                            "box": "check_box_outline_blank",
                            "sphere": "radio_button_unchecked",
                            "import": "input"}[s["kind"]]}
                  for s in self._solids]
        self.tree.ui_nodes = [
            {"id": "model", "label": "Model", "icon": "widgets", "children": [
                {"id": "components", "label": "Components",
                 "icon": "category", "children": solids},
                {"id": TAB_MESH, "label": "Mesh", "icon": "grid_on"},
            ]},
            {"id": "results", "label": "Results", "icon": "insights",
             "children": [
                 {"id": TAB_FIELD, "label": "3D field (ROM)", "icon": "bolt"},
                 {"id": TAB_S, "label": "S-parameters", "icon": "timeline"},
                 {"id": TAB_Z, "label": "Z-parameters", "icon": "timeline"},
             ]},
        ]
        # ui_default_expand_all only acts on the FIRST render — nodes added
        # later would appear collapsed (invisible) without this.
        self.tree.ui_expanded = ["model", "components", "results"]
        self.bc_solid.ui_options = [s["name"] for s in self._solids
                                    if s["kind"] != "import"]

    def _on_tree_select(self, e):
        key = e.value
        if not key:
            return
        if key.startswith("solid:"):
            self._selected = key.split(":", 1)[1]
            self._show_solid(self._selected)
            self.bc_solid.ui_model_value = self._selected
            self._bc_refresh_faces()
        elif key in (TAB_GEO, TAB_MESH, TAB_FIELD, TAB_S, TAB_Z):
            self._set_tab(key)

    def _build_viewport(self):
        self.webgui_geo = _fill_webgui(WebguiComponent(id="vp_geo"))
        self.webgui_mesh = _fill_webgui(WebguiComponent(id="vp_mesh"))
        self.webgui_field = _fill_webgui(WebguiComponent(id="vp_field"))
        # Click a face in the 3D view -> select it in the BC editor.
        self.webgui_geo.on_click(self._on_pick)
        self.webgui_mesh.on_click(self._on_pick)
        self.plot_s = PlotlyComponent(id="vp_s", ui_style=_FILL)
        self.plot_z = PlotlyComponent(id="vp_z", ui_style=_FILL)

        # Right-click menu on the plot wrappers (menus are SIBLINGS of the
        # plot — never children, or Plotly.react wipes them from the DOM).
        def plot_wrap(plot, kind):
            menu = QMenu(ui_context_menu=True, ui_auto_close=True)
            lst = QList(ui_style="min-width: 170px")
            reset = QItem(QItemSection("Reset zoom / redraw"),
                          ui_clickable=True)
            reset.on("click", lambda: self._redraw_1d())
            csv = QItem(QItemSection("Export data (CSV)"), ui_clickable=True)
            csv.on("click", lambda: self._export_csv())
            lst.ui_children = [reset, csv]
            menu.ui_children = [lst]
            return Div(plot, menu, ui_style=_FILL)

        self.tabs = QTabs(
            QTab(ui_name=TAB_GEO, ui_label="Geometry", ui_icon="category"),
            QTab(ui_name=TAB_MESH, ui_label="Mesh", ui_icon="grid_on"),
            QTab(ui_name=TAB_FIELD, ui_label="3D field", ui_icon="bolt"),
            QTab(ui_name=TAB_S, ui_label="S", ui_icon="timeline"),
            QTab(ui_name=TAB_Z, ui_label="Z", ui_icon="timeline"),
            ui_model_value=TAB_GEO, ui_dense=True, ui_no_caps=True,
            ui_align="left", ui_class="bg-grey-2 text-primary",
        )
        self.panels = QTabPanels(
            QTabPanel(self.webgui_geo, ui_name=TAB_GEO, ui_class="q-pa-none",
                      ui_style="height: 100%"),
            QTabPanel(self.webgui_mesh, ui_name=TAB_MESH,
                      ui_class="q-pa-none", ui_style="height: 100%"),
            QTabPanel(self.webgui_field, ui_name=TAB_FIELD,
                      ui_class="q-pa-none", ui_style="height: 100%"),
            QTabPanel(plot_wrap(self.plot_s, "s"), ui_name=TAB_S,
                      ui_class="q-pa-none", ui_style="height: 100%"),
            QTabPanel(plot_wrap(self.plot_z, "z"), ui_name=TAB_Z,
                      ui_class="q-pa-none", ui_style="height: 100%"),
            ui_model_value=TAB_GEO,
            ui_style="flex: 1 1 auto; min-height: 0",
        )
        self.tabs.on_update_model_value(lambda e: self._set_tab(e.value))
        self._viewport = Div(self.tabs, self.panels,
                             ui_class="column no-wrap", ui_style=_FILL)

    def _set_tab(self, name):
        if self.tabs.ui_model_value != name:
            self.tabs.ui_model_value = name
        self.panels.ui_model_value = name
        # Hidden panels render at zero size — re-render on focus.
        if name == TAB_S and self._fig_s is not None:
            self.plot_s.draw(self._fig_s)
        elif name == TAB_Z and self._fig_z is not None:
            self.plot_z.draw(self._fig_z)
        elif name == TAB_FIELD and self._concat is not None:
            self._draw_field()

    def _tree_panel(self):
        # Right-click anywhere in the tree: context menu acting on the
        # selected solid (left-click a component first to select it).
        menu = QMenu(ui_context_menu=True, ui_auto_close=True)
        lst = QList(ui_style="min-width: 170px")

        def item(icon, text, fn):
            it = QItem(QItemSection(QIcon(ui_name=icon), ui_avatar=True),
                       QItemSection(text), ui_clickable=True, ui_dense=True)
            it.on("click", fn)
            return it

        lst.ui_children = [
            item("drive_file_rename_outline", "Rename solid…",
                 self._open_rename),
            item("delete", "Delete solid", self._delete_selected),
            item("delete_sweep", "Clear all solids", self._clear_solids),
        ]
        menu.ui_children = [lst]

        # rename dialog
        self.rn_name = _txt("new name", "", 180)
        rn_btn = QBtn(ui_label="Rename", ui_color="primary",
                      ui_class="q-ma-sm")
        rn_btn.on("click", self._do_rename)
        self.dlg_rename = QDialog(QCard(
            QCardSection(Heading("Rename solid", 6)),
            QCardSection(self.rn_name), QCardSection(rn_btn),
            ui_style="min-width: 300px"))
        self._dialogs.append(self.dlg_rename)

        return Div(
            QToolbar(QIcon(ui_name="account_tree"),
                     Label("Project", ui_class="q-ml-sm text-weight-bold"),
                     ui_class="bg-grey-2"),
            self.tree,
            menu,
            ui_class="column no-wrap",
            ui_style="height: 100%; overflow: auto",
        )

    def _open_rename(self):
        if not self._selected:
            self._log("Select a solid in the tree first (left-click), then "
                      "right-click → Rename.")
            return
        self.rn_name.ui_model_value = self._selected
        self.dlg_rename.ui_model_value = True

    def _do_rename(self):
        try:
            old, new = self._selected, self.rn_name.ui_model_value
            if not (old and new):
                raise ValueError("select a solid and give a new name")
            if any(s["name"] == new for s in self._solids):
                raise ValueError(f"a solid named '{new}' already exists")
            for s in self._solids:
                if s["name"] == old:
                    s["name"] = new
            self._selected = new
            self._refresh_tree()
            self.dlg_rename.ui_model_value = False
            self._log(f"Renamed solid '{old}' -> '{new}'.")
        except Exception as e:
            self._err(e)

    # ================================================================== #
    # messages (expandable upwards-readable log)
    # ================================================================== #
    def _build_messages(self):
        """Status line + log, in the lower pane of a draggable splitter."""
        self.status = Label("", ui_class="text-weight-medium q-ml-sm")
        self.log_div = Div(ui_style="white-space: pre-wrap; font-family: "
                                    "monospace; font-size: 12px; overflow: "
                                    "auto; padding: 2px 8px; flex: 1 1 auto; "
                                    "min-height: 0")
        self._bottom = Div(
            Row(QIcon(ui_name="drag_handle", ui_size="xs"),
                QIcon(ui_name="info", ui_size="xs"), self.status,
                ui_class="items-center q-gutter-x-xs q-px-sm bg-grey-2",
                ui_style="min-height: 26px"),
            self.log_div,
            ui_class="column no-wrap", ui_style=_FILL,
        )

    def _log(self, msg, error=False):
        stamp = time.strftime("%H:%M:%S")
        self._messages.append(f"[{stamp}] {msg}")
        self._messages = self._messages[-300:]
        self.log_div.ui_children = ["\n".join(reversed(self._messages))]
        self.status.text = ("ERROR — open Messages below" if error and
                            len(msg) > 120 else msg.splitlines()[0][:160])

    def _err(self, e):
        self._log(f"{e}\n{traceback.format_exc()}", error=True)

    # ================================================================== #
    # geometry actions
    # ================================================================== #
    def _mk_geo(self, kind):
        from cavsim3d.geometry.primitives import (
            Box, CircularWaveguide, RectangularWaveguide, Sphere)
        if kind == "rwg":
            d = self.rw
            g = RectangularWaveguide(a=d["a"].ui_model_value,
                                     b=d["b"].ui_model_value,
                                     L=d["L"].ui_model_value,
                                     maxh=d["maxh"].ui_model_value)
        elif kind == "cyl":
            d = self.cy
            g = CircularWaveguide(radius=d["r"].ui_model_value,
                                  length=d["L"].ui_model_value,
                                  maxh=d["maxh"].ui_model_value)
        elif kind == "box":
            d = self.bx
            g = Box(dimensions=(d["a"].ui_model_value, d["b"].ui_model_value,
                                d["L"].ui_model_value),
                    maxh=d["maxh"].ui_model_value)
            from netgen.occ import X, Y, Z
            g.geo.faces.Min(Z).name = d["f_zmin"].ui_model_value
            g.geo.faces.Max(Z).name = d["f_zmax"].ui_model_value
            g.geo.faces.Min(Y).name = d["f_ymin"].ui_model_value
            g.geo.faces.Max(Y).name = d["f_ymax"].ui_model_value
            g.geo.faces.Min(X).name = d["f_xmin"].ui_model_value
            g.geo.faces.Max(X).name = d["f_xmax"].ui_model_value
            walls = [d[k].ui_model_value for k in
                     ("f_ymin", "f_ymax", "f_xmin", "f_xmax")]
            g.bc = "|".join(walls)
        elif kind == "sphere":
            d = self.sp
            g = Sphere(radius=d["r"].ui_model_value,
                       maxh=d["maxh"].ui_model_value,
                       material=d["mat"].ui_model_value or "vacuum")
        else:
            raise ValueError(kind)
        # material + (re)mesh so renames/materials take effect
        mat = d.get("mat")
        if mat is not None and kind != "sphere":
            g.geo.mat(mat.ui_model_value or "vacuum")
        g.generate_mesh(maxh=d["maxh"].ui_model_value)
        return g, d

    def _add_solid(self, kind):
        try:
            if kind == "import":
                d = self.im
                path = Path(d["path"].ui_model_value)
                if not path.exists():
                    raise FileNotFoundError(f"project not found: {path}")
                entry = dict(name=d["name"].ui_model_value, kind=kind,
                             path=str(path), geo=None,
                             n=int(d["n"].ui_model_value))
            else:
                g, d = self._mk_geo(kind)
                entry = dict(name=d["name"].ui_model_value, kind=kind, geo=g,
                             n=int(d.get("n", _num("", 1)).ui_model_value
                                   if "n" in d else 1),
                             maxh=d["maxh"].ui_model_value if "maxh" in d
                             else 0.05)
            if any(s["name"] == entry["name"] for s in self._solids):
                raise ValueError(f"a solid named '{entry['name']}' exists")
            self._solids.append(entry)
            self._selected = entry["name"]
            self._refresh_tree()
            for dlg in self._dialogs:
                dlg.ui_model_value = False
            self._show_solid(entry["name"])
            self._log(f"Added {kind} '{entry['name']}'"
                      + (f" ×{entry['n']}" if entry["n"] > 1 else "")
                      + (f" from {entry.get('path')}" if kind == "import"
                         else f" — mesh {entry['geo'].mesh.ne} elements"))
        except Exception as e:
            self._err(e)

    def _show_solid(self, name):
        s = next((x for x in self._solids if x["name"] == name), None)
        if s is None:
            return
        if s["kind"] == "import":
            self._log(f"'{name}' is an imported model — no live geometry "
                      f"(source: {s['path']}).")
            return
        try:
            self.webgui_geo.draw(s["geo"].geo)
            self.webgui_mesh.draw(s["geo"].mesh)
            self._set_tab(TAB_GEO)
        except Exception as e:
            self._err(e)

    def _delete_selected(self):
        if not self._selected:
            self._log("Select a solid in the tree first.")
            return
        self._solids = [s for s in self._solids if s["name"] != self._selected]
        self._log(f"Deleted '{self._selected}'.")
        self._selected = None
        self._refresh_tree()

    def _clear_solids(self):
        self._solids.clear()
        self._selected = None
        self._refresh_tree()
        self._log("Cleared all solids.")

    # ================================================================== #
    # meshing / boundary conditions
    # ================================================================== #
    def _on_generate_mesh(self):
        try:
            if not self._solids:
                raise ValueError("no solids — add geometry first")
            if self.tgl_glue.ui_model_value:
                asm = self._make_assembly(None)
                asm.generate_mesh(maxh=self.in_glue_maxh.ui_model_value)
                self.webgui_mesh.draw(asm.mesh)
                self.lbl_mesh.text = f"glued mesh: {asm.mesh.ne} elements, " \
                                     f"{len(set(asm.mesh.GetMaterials()))} domains"
            else:
                name = self._selected or self._solids[-1]["name"]
                s = next(x for x in self._solids if x["name"] == name)
                if s["kind"] == "import":
                    raise ValueError("imported models keep their own mesh")
                s["geo"].generate_mesh(maxh=s.get("maxh", 0.05))
                self.webgui_mesh.draw(s["geo"].mesh)
                self.lbl_mesh.text = f"'{name}': {s['geo'].mesh.ne} elements"
            self._set_tab(TAB_MESH)
            self._log(self.lbl_mesh.text)
        except Exception as e:
            self._err(e)

    def _on_pick(self, e):
        """Click in a 3D view: resolve the picked face and select it in the
        BC editor (payload shape depends on the webgui frontend — anything
        unrecognized is logged so it can be wired up)."""
        try:
            v = e.value if hasattr(e, "value") else e
            face = None
            if isinstance(v, dict):
                # common shapes: {'dim': 2, 'index': i} or a named region
                for key in ("name", "bnd_name", "region", "boundary"):
                    if isinstance(v.get(key), str):
                        face = v[key]
                        break
                if face is None and v.get("dim") == 2 and "index" in v:
                    name = self._selected or (self._solids[-1]["name"]
                                              if self._solids else None)
                    s = next((x for x in self._solids if x["name"] == name),
                             None)
                    if s is not None and s["kind"] != "import":
                        bnds = list(s["geo"].mesh.GetBoundaries())
                        idx = int(v["index"])
                        if 0 <= idx < len(bnds):
                            face = bnds[idx]
            if face:
                self.bc_face.ui_model_value = face
                self.mode_tabs.ui_model_value = MODE_BC
                self.mode_panels.ui_model_value = MODE_BC
                self._log(f"Picked face '{face}' — edit it in the Boundary "
                          "conditions ribbon.")
            else:
                self._log(f"3D pick payload: {v!r:.300}")
        except Exception as ex:
            self._err(ex)

    def _bc_refresh_faces(self):
        name = self.bc_solid.ui_model_value
        s = next((x for x in self._solids if x["name"] == name), None)
        if s is None or s["kind"] == "import":
            self.bc_face.ui_options = []
            return
        faces = sorted({f.name or "(unnamed)" for f in s["geo"].geo.faces})
        self.bc_face.ui_options = faces

    def _bc_rename(self):
        try:
            name = self.bc_solid.ui_model_value
            old = self.bc_face.ui_model_value
            new = self.bc_newname.ui_model_value
            if not (name and old and new):
                raise ValueError("pick a solid + face and give a new name")
            s = next(x for x in self._solids if x["name"] == name)
            n = 0
            for f in s["geo"].geo.faces:
                if (f.name or "(unnamed)") == old:
                    f.name = new
                    n += 1
            s["geo"].generate_mesh(maxh=s.get("maxh", 0.05))
            self._bc_refresh_faces()
            self.webgui_mesh.draw(s["geo"].mesh)
            self._log(f"Renamed {n} face(s) '{old}' -> '{new}' on '{name}' "
                      "and re-meshed.")
        except Exception as e:
            self._err(e)

    def _bc_material(self):
        try:
            name = self.bc_solid.ui_model_value
            if not name:
                raise ValueError("pick a solid")
            s = next(x for x in self._solids if x["name"] == name)
            s["geo"].geo.mat(self.bc_mat.ui_model_value or "vacuum")
            s["geo"].generate_mesh(maxh=s.get("maxh", 0.05))
            self._log(f"Material of '{name}' set to "
                      f"'{self.bc_mat.ui_model_value}' (re-meshed).")
        except Exception as e:
            self._err(e)

    # ================================================================== #
    # simulation
    # ================================================================== #
    def on_run(self):
        if self._busy:
            self._log("A run is already in progress.")
            return
        if not self._solids:
            self._log("Add at least one solid first (Geometry ribbon).")
            return
        if self.tgl_async.ui_model_value:
            threading.Thread(target=self._run_guarded, daemon=True).start()
        else:
            self._run_guarded()

    def _run_guarded(self):
        self._busy = True
        self.btn_run.ui_loading = True
        try:
            self._run_pipeline()
        except Exception as e:
            self._err(e)
        finally:
            self._busy = False
            self.btn_run.ui_loading = False

    def _make_assembly(self, proj):
        """Build the passive netlist from the solid list (order = tree order)."""
        from cavsim3d.geometry.assembly import Assembly
        asm = proj.create_assembly(main_axis="Z") if proj is not None \
            else Assembly(main_axis="Z")
        prev = None
        for s in self._solids:
            comp = s["geo"]
            if s["kind"] == "import":
                comp = proj.fds.import_model(s["path"]) if proj is not None \
                    else s["path"]
            kw = dict(n=s["n"]) if s["n"] > 1 else {}
            if prev is None:
                asm.add(s["name"], comp, **kw)
            else:
                asm.add(s["name"], comp, after=prev, **kw)
            prev = s["name"]
        return asm

    def _run_pipeline(self):
        from cavsim3d.core.em_project import EMProject

        cfg = dict(fmin=self.in_fmin.ui_model_value,
                   fmax=self.in_fmax.ui_model_value,
                   nsamples=int(self.in_nsamp.ui_model_value),
                   order=int(self.in_order.ui_model_value),
                   nportmodes=int(self.in_modes.ui_model_value),
                   store_snapshots=True)
        tol = self.in_tol.ui_model_value
        nsweep = int(self.in_sweep.ui_model_value)
        glue = self.tgl_glue.ui_model_value
        RUNS_DIR.mkdir(exist_ok=True)
        t = {}

        # Netlist when the assembly carries imports or repeat counts (the
        # core's rule); several plain solids couple through ONE glued
        # conformal mesh; exactly one plain solid runs fds.fom.rom.
        has_import = any(s["kind"] == "import" for s in self._solids)
        netlist = has_import or any(s["n"] > 1 for s in self._solids)
        single = len(self._solids) == 1 and not netlist
        if netlist and glue:
            self._log("Note: imports/repeat counts imply a NETLIST — the "
                      "glue toggle is ignored (sections keep their meshes).")
        if not netlist and not single and not glue:
            self._log("Note: several plain solids couple through one glued "
                      "conformal mesh — gluing automatically.")
            glue = True

        proj = EMProject(name=self.in_name.ui_model_value or "bench",
                         base_dir=str(RUNS_DIR), overwrite=True)
        t0 = time.perf_counter()

        if single:
            self._log("Single solid: fds.solve -> fom.reduce -> rom sweep …")
            proj.geometry = self._solids[0]["geo"]
            proj.fds.solve(config=cfg)
            t["FOM"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            rom = proj.fds.fom.reduce(tol=tol)
            t["ROM"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            res = rom.solve(fmin=cfg["fmin"], fmax=cfg["fmax"],
                            nsamples=nsweep)
            t["sweep"] = time.perf_counter() - t0
            self._concat = None
            self._freqs = np.linspace(cfg["fmin"], cfg["fmax"], nsweep) * 1e9
        else:
            if not netlist:
                self._log("Glued multi-solid: one conformal mesh …")
                asm = self._make_assembly(proj)
                asm.generate_mesh(maxh=self.in_glue_maxh.ui_model_value)
                proj.fds.solve(config=dict(**cfg, per_domain=True,
                                           global_method=None))
            else:
                self._log("Netlist: FOM per unique section …")
                self._make_assembly(proj)
                proj.fds.solve(config=cfg)
            t["FOM"] = time.perf_counter() - t0

            self._log("ROM stage …")
            t0 = time.perf_counter()
            roms = proj.fds.foms.reduce(tol=tol)
            t["ROM"] = time.perf_counter() - t0

            self._log("Concatenating …")
            t0 = time.perf_counter()
            concat = roms.concatenate()
            t["concat"] = time.perf_counter() - t0

            self._log(f"Sweeping {nsweep} points …")
            t0 = time.perf_counter()
            res = concat.solve(config=dict(fmin=cfg["fmin"], fmax=cfg["fmax"],
                                           nsamples=nsweep))
            t["sweep"] = time.perf_counter() - t0

            if self.tgl_concat_rom.ui_model_value:
                t0 = time.perf_counter()
                concat.reduce(tol=tol * 0.1)
                t["concat.rom"] = time.perf_counter() - t0
            self._concat = concat
            self._freqs = concat.frequencies

        self._proj = proj
        self._S, self._Z = res.get("S"), res.get("Z")
        self._Sa = self._analytical_reference()

        self._redraw_1d()
        self._sync_field_controls()
        self._draw_field()
        self._set_tab(TAB_S)
        self.mode_tabs.ui_model_value = MODE_RES
        self.mode_panels.ui_model_value = MODE_RES

        timing = "  ".join(f"{k}={v:.2f}s" for k, v in t.items())
        extra = ""
        if self._Sa is not None and self._S is not None:
            dmag = float(np.max(np.abs(np.abs(self._S[:, 1, 0])
                                       - np.abs(self._Sa["S21"]))))
            extra = f" | max||S21|-analytical|={dmag:.4f}"
        self._log(f"DONE ({timing}){extra} — project: {proj.project_path}")

    def _analytical_reference(self):
        """|S| reference for a pure rectangular-waveguide chain, else None."""
        from cavsim3d.analytical.rectangular_waveguide import RWGAnalytical
        live = [s for s in self._solids if s["kind"] == "rwg"]
        if len(live) != len(self._solids) or not live or self._freqs is None:
            return None
        a0, b0 = live[0]["geo"].a, live[0]["geo"].b
        if any(abs(s["geo"].a - a0) > 1e-12 or abs(s["geo"].b - b0) > 1e-12
               for s in live):
            return None
        L = sum(s["geo"].L * s["n"] for s in live)
        return RWGAnalytical(a=a0, L=L, b=b0).s_parameters(
            self._freqs / 1e9, Z0_ref="ZTE")

    # ================================================================== #
    # results
    # ================================================================== #
    def _redraw_1d(self):
        if self._S is None and self._Z is None:
            return
        import plotly.graph_objects as go
        f = self._freqs / 1e9

        if self._S is not None:
            fig = go.Figure()
            npo = self._S.shape[1]
            for i in range(npo):
                for j in range(npo):
                    fig.add_scatter(x=f, y=np.abs(self._S[:, i, j]),
                                    name=f"|S{i+1}{j+1}|")
            if self._Sa is not None:
                fig.add_scatter(x=f, y=np.abs(self._Sa["S21"]),
                                name="|S21| analytical",
                                line=dict(dash="dash", color="black"))
                fig.add_scatter(x=f, y=np.abs(self._Sa["S11"]),
                                name="|S11| analytical",
                                line=dict(dash="dot", color="gray"))
            fig.update_layout(xaxis_title="f [GHz]", yaxis_title="|S|",
                              margin=dict(l=50, r=10, t=25, b=40),
                              legend=dict(orientation="h"))
            self._fig_s = fig
            self.plot_s.draw(fig)

        if self._Z is not None:
            fig = go.Figure()
            npo = self._Z.shape[1]
            for i in range(npo):
                for j in range(npo):
                    fig.add_scatter(x=f, y=np.abs(self._Z[:, i, j]),
                                    name=f"|Z{i+1}{j+1}|")
            fig.update_layout(xaxis_title="f [GHz]", yaxis_title="|Z| [Ω]",
                              margin=dict(l=50, r=10, t=25, b=40),
                              legend=dict(orientation="h"))
            self._fig_z = fig
            self.plot_z.draw(fig)

    def _sync_field_controls(self):
        if self._concat is None:
            self.sel_section.ui_options = []
            self.sel_port.ui_options = []
            self.sl_freq.ui_max = 0
            return
        c = self._concat
        self.sel_section.ui_options = [
            {"label": s.domain, "value": i} for i, s in enumerate(c.structures)]
        self.sel_section.ui_model_value = 0
        self.sel_port.ui_options = list(c.ports)
        self.sel_port.ui_model_value = c.ports[0] if c.ports else None
        self.sl_freq.ui_max = max(0, len(self._freqs) - 1)
        self.sl_freq.ui_model_value = len(self._freqs) // 2

    def _draw_field(self):
        if self._concat is None:
            self._log("3D field needs a concatenated run (netlist / glued "
                      "multi-solid).")
            return
        try:
            sec = self.sel_section.ui_model_value
            if isinstance(sec, dict):
                sec = sec.get("value", 0)
            sec = 0 if sec is None else int(sec)
            fidx = int(self.sl_freq.ui_model_value or 0)
            cf, mesh, label = self._concat.reconstruct_section_field(
                section_idx=sec, freq_idx=fidx,
                excitation_port=self.sel_port.ui_model_value
                or self._concat.ports[0],
                field_type=self.sel_field.ui_model_value or "E",
                component=self.sel_comp.ui_model_value or "abs")
            self.webgui_field.draw(cf, mesh, label)
            self.lbl_freq.text = f"f = {self._freqs[fidx] / 1e9:.4f} GHz"
        except Exception as e:
            self._err(e)

    def _export_csv(self):
        try:
            if self._S is None or self._proj is None:
                raise ValueError("run the pipeline first")
            out = Path(self._proj.project_path) / "exports"
            out.mkdir(exist_ok=True)
            f = self._freqs / 1e9
            npo = self._S.shape[1]
            head = "f_GHz," + ",".join(
                f"reS{i+1}{j+1},imS{i+1}{j+1}"
                for i in range(npo) for j in range(npo))
            rows = [head]
            for k in range(len(f)):
                cells = [f"{f[k]:.6f}"]
                for i in range(npo):
                    for j in range(npo):
                        cells += [f"{self._S[k, i, j].real:.8e}",
                                  f"{self._S[k, i, j].imag:.8e}"]
                rows.append(",".join(cells))
            (out / "s_params.csv").write_text("\n".join(rows))
            self._log(f"Exported {out / 's_params.csv'}")
        except Exception as e:
            self._err(e)

    def _reveal_folder(self):
        self._log(f"Project folder: {self._proj.project_path}"
                  if self._proj else "No run yet — no project folder.")
