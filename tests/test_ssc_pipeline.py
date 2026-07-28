"""State-space concatenation through the standard EMProject pipeline.

The operation philosophy under test (no separate chain/driver class, and the
user never touches ConcatenatedSystem directly — ``concat`` is the object
RETURNED by ``concatenate()`` on a foms/roms collection):

    proj = EMProject(...)
    asm = proj.create_assembly('Z')        # passive netlist
    asm.add('cell', geo, n=3)              # repeat count, computed once
    asm.add('hom', proj.fds.import_model(path), after='cell')
    proj.fds.solve(config=...)             # FOM   (stage 1, per unique comp.)
    roms = proj.fds.foms.reduce(tol)       # ROM   (stage 2, auto-saved)
    concat = roms.concatenate()            # Concat (stage 3)
    concat.reduce(tol)                     # further ROM (optional stage 4)

``ConcatenatedSystem.from_flat_roms`` is INTERNAL plumbing used by
``roms.concatenate()``; the user never touches ConcatenatedSystem.
"""

import numpy as np
import pytest

# Import the solver chain first so the rom<->solvers modules finish
# initialising (avoids a circular import when run standalone).
from cavsim3d.core.em_project import EMProject  # noqa: F401
from cavsim3d.geometry.primitives import RectangularWaveguide
from cavsim3d.analytical.rectangular_waveguide import RWGAnalytical

A = 0.1
SEC_L = 0.06667
MAXH = 0.06
CFG = dict(fmin=1.8, fmax=2.4, nsamples=4, nportmodes=1)


def _section():
    return RectangularWaveguide(a=A, L=SEC_L, maxh=MAXH)


CFG_ORDER = dict(**CFG, order=2)


# ===========================================================================
# THE user-facing pipeline: proj.fds.solve -> foms.reduce -> concatenate
# ===========================================================================

class TestProjPipeline:

    def test_repeat_n_full_pipeline(self, tmp_path):
        proj = EMProject(name='module', base_dir=str(tmp_path), overwrite=True)
        asm = proj.create_assembly(main_axis='Z')      # still the entry point
        asm.add('cell', _section(), n=3)

        proj.fds.solve(config=CFG_ORDER)               # FOM stage
        roms = proj.fds.foms.reduce(tol=1e-9)          # ROM stage
        concat = roms.concatenate()                    # concat stage
        assert len(concat.structures) == 3
        assert concat.n_external_ports == 2

        res = concat.solve(config=dict(fmin=1.8, fmax=2.4, nsamples=5))
        S = res['S']
        f = concat.frequencies
        ana = RWGAnalytical(a=A, L=3 * SEC_L, b=A / 2)
        Sa = ana.s_parameters(f / 1e9, Z0_ref='ZTE')
        assert np.all(np.abs(S[:, 0, 0]) < 0.05)
        dphase = np.angle(S[:, 1, 0]) - np.angle(Sa['S21'])
        dphase = (dphase + np.pi) % (2 * np.pi) - np.pi
        assert np.all(np.abs(np.degrees(dphase)) < 25)

        # STANDARD project layout (identical to a multi-solid project): ONE
        # fds; sections distinguished ONLY by the filename suffix convention;
        # a single mesh/ and geometry/ folder at the project top level.
        module = tmp_path / 'module'
        foms = module / 'fds' / 'foms'
        assert (foms / 'matrices' / 'K_cell.h5').exists()
        assert (foms / 's' / 's_cell.h5').exists()
        assert (foms / 'roms' / 'matrices' / 'A_r_cell.h5').exists()
        assert (foms / 'roms' / 'structures.json').exists()
        assert (foms / 'roms' / 'concat').exists()
        assert (module / 'mesh' / 'mesh_cell.pkl').exists()
        assert (module / 'geometry' / 'components' / 'cell.step').exists()
        assert (module / 'fds' / 'config.json').exists()
        assert (module / 'project.json').exists()
        # foms/roms may hold ONLY the canonical folders (no per-section dirs).
        allowed_foms = {'matrices', 'eigenmodes', 's', 'z', 'snapshots', 'roms'}
        assert {p.name for p in foms.iterdir() if p.is_dir()} <= allowed_foms
        allowed_roms = {'matrices', 'eigenmodes', 's', 'z', 'snapshots',
                        'concat', 'rom', 'roms'}
        assert {p.name for p in (foms / 'roms').iterdir()
                if p.is_dir()} <= allowed_roms
        # NO nested sub-project (single fds per project).
        assert not any(p.name == 'fds' for p in foms.rglob('fds'))
        assert not (module / 'components').exists()

        # further reduction of the coupled system (concat.rom stage)
        assert concat.reduce(tol=1e-10) is not None

    def test_import_model_mixed_netlist(self, tmp_path):
        # An earlier campaign: plain single-solid project, solved + reduced.
        prev = EMProject(name='earlier', base_dir=str(tmp_path), overwrite=True)
        prev.geometry = _section()
        prev.fds.solve(config=CFG_ORDER)
        prev.fds.fom.reduce(tol=1e-9)

        # New analysis mixes an imported model with fresh geometry.
        proj = EMProject(name='mixed', base_dir=str(tmp_path), overwrite=True)
        asm = proj.create_assembly(main_axis='Z')
        legacy = proj.fds.import_model(str(tmp_path / 'earlier'))
        assert 'earlier' in repr(legacy) and legacy.training_band is not None
        asm.add('fresh', _section())
        asm.add('legacy', legacy, after='fresh')

        proj.fds.solve(config=CFG_ORDER)               # runs fresh, loads legacy
        concat = proj.fds.foms.reduce(tol=1e-9).concatenate()
        assert len(concat.structures) == 2
        res = concat.solve(config=dict(fmin=1.8, fmax=2.4, nsamples=5))
        assert np.allclose(np.abs(res['S'][:, 1, 0]), 1.0, atol=0.05)

    def test_import_model_without_results_raises(self, tmp_path):
        proj = EMProject(name='p', base_dir=str(tmp_path), overwrite=True)
        proj.create_assembly(main_axis='Z')
        empty = tmp_path / 'never_run'
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            proj.fds.import_model(str(empty))

    def test_foms_before_solve_raises_helpfully(self, tmp_path):
        proj = EMProject(name='p2', base_dir=str(tmp_path), overwrite=True)
        asm = proj.create_assembly(main_axis='Z')
        asm.add('cell', _section(), n=2)
        with pytest.raises(RuntimeError, match='netlist'):
            _ = proj.fds.foms

    def test_netlist_fom_level_concat_not_supported(self, tmp_path):
        proj = EMProject(name='p3', base_dir=str(tmp_path), overwrite=True)
        asm = proj.create_assembly(main_axis='Z')
        asm.add('cell', _section(), n=2)
        proj.fds.solve(config=CFG_ORDER)
        with pytest.raises(NotImplementedError):
            proj.fds.foms.concatenate()


# ===========================================================================
# Imported sections are COPIED (renamed) into the flat foms tree, not linked
# ===========================================================================

class TestImportCopy:

    def _module_with_import(self, tmp_path):
        prev = EMProject(name='earlier', base_dir=str(tmp_path), overwrite=True)
        prev.geometry = _section()
        prev.fds.solve(config=dict(CFG_ORDER, store_snapshots=True))
        prev.fds.fom.reduce(tol=1e-9)
        prev.save()

        proj = EMProject(name='module', base_dir=str(tmp_path), overwrite=True)
        asm = proj.create_assembly(main_axis='Z')
        asm.add('fresh', _section())
        asm.add('legacy', proj.fds.import_model(str(tmp_path / 'earlier')),
                after='fresh')
        proj.fds.solve(config=dict(CFG_ORDER, store_snapshots=True))
        proj.fds.foms.reduce(tol=1e-9).concatenate()
        return tmp_path / 'module', tmp_path / 'earlier'

    def test_imported_section_indistinguishable_from_computed(self, tmp_path):
        module, _src = self._module_with_import(tmp_path)
        foms = module / 'fds' / 'foms'
        # Imported ('legacy') and computed ('fresh') sections leave IDENTICAL
        # footprints: suffix-renamed files in the same shared folders.
        for sec in ('fresh', 'legacy'):
            assert (foms / 'matrices' / f'K_{sec}.h5').exists()
            assert (foms / 'roms' / 'matrices' / f'A_r_{sec}.h5').exists()
            assert (module / 'mesh' / f'mesh_{sec}.pkl').exists()
            assert (module / 'geometry' / 'components' / f'{sec}.step').exists()
            assert not (foms / sec).exists()      # no per-section folders

    def test_module_is_self_contained_after_source_deleted(self, tmp_path):
        module, src = self._module_with_import(tmp_path)
        import shutil
        shutil.rmtree(src)                       # source gone
        # The module reloads and its coupled results stand on their own.
        proj = EMProject(name='module', base_dir=str(tmp_path))
        assert proj is not None
        assert (module / 'fds' / 'foms' / 'roms' / 'concat').exists()

    def test_missing_source_at_solve_errors(self, tmp_path):
        proj = EMProject(name='m2', base_dir=str(tmp_path), overwrite=True)
        asm = proj.create_assembly(main_axis='Z')
        asm.add('fresh', _section())
        asm._components['ghost'] = asm._components['fresh']  # placeholder
        # Point a section at a non-existent project.
        proj2 = EMProject(name='m3', base_dir=str(tmp_path), overwrite=True)
        a2 = proj2.create_assembly(main_axis='Z')
        a2.add('fresh', _section())
        a2.add('missing', str(tmp_path / 'does_not_exist'), after='fresh')
        with pytest.raises(FileNotFoundError):
            proj2.fds.solve(config=CFG_ORDER)


# ===========================================================================
# Netlist defaults
# ===========================================================================

class TestNetlistDefaults:

    def test_default_n_is_one(self, tmp_path):
        proj = EMProject(name='single_cell', base_dir=str(tmp_path),
                         overwrite=True)
        asm = proj.create_assembly(main_axis='Z')
        # A single imported reference (no n given) is still a netlist.
        prev = EMProject(name='earlier', base_dir=str(tmp_path), overwrite=True)
        prev.geometry = _section()
        prev.fds.solve(config=dict(CFG_ORDER, store_snapshots=True))
        prev.fds.fom.reduce(tol=1e-9)
        prev.save()
        asm.add('cell', proj.fds.import_model(str(tmp_path / 'earlier')))
        proj.fds.solve(config=CFG_ORDER)
        concat = proj.fds.foms.reduce(tol=1e-9).concatenate()
        assert len(concat.structures) == 1
        assert concat.n_external_ports == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
