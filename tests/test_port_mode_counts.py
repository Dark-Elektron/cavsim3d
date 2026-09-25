"""Per-port mode counts: ``nportmodes`` as an int, a list or a dict.

CST assigns modes per port (e.g. 1 on a coax, 3 on a beam pipe). A single
global count could not express that, so an asymmetric CST model was not
reproducible.
"""

import pytest

# Import the solver chain first so the rom<->solvers modules finish
# initialising (avoids a circular import when run standalone).
from cavsim3d.core.em_project import EMProject  # noqa: F401
from cavsim3d.geometry.primitives import RectangularWaveguide
from cavsim3d.solvers.ports import resolve_port_mode_counts

PORTS = ['port1', 'port2', 'port3', 'port4']


class TestResolvePortModeCounts:
    """Normalisation of the three accepted spellings."""

    def test_int_applies_to_every_port(self):
        assert resolve_port_mode_counts(3, PORTS) == {p: 3 for p in PORTS}

    def test_none_defaults_to_one(self):
        assert resolve_port_mode_counts(None, PORTS) == {p: 1 for p in PORTS}

    def test_list_is_positional(self):
        assert resolve_port_mode_counts([1, 3, 1, 2], PORTS) == {
            'port1': 1, 'port2': 3, 'port3': 1, 'port4': 2}

    def test_tuple_accepted_like_a_list(self):
        assert resolve_port_mode_counts((2, 2, 2, 2), PORTS) == {
            p: 2 for p in PORTS}

    def test_dict_fills_missing_from_default(self):
        assert resolve_port_mode_counts({'port2': 3, 'default': 1}, PORTS) == {
            'port1': 1, 'port2': 3, 'port3': 1, 'port4': 1}

    def test_dict_without_default_fills_with_one(self):
        assert resolve_port_mode_counts({'port4': 5}, PORTS)['port1'] == 1
        assert resolve_port_mode_counts({'port4': 5}, PORTS)['port4'] == 5


class TestValidation:
    """A wrong spec must fail loudly, naming the valid ports."""

    def test_short_list_rejected(self):
        with pytest.raises(ValueError, match=r"2 entries but the model has 4"):
            resolve_port_mode_counts([1, 2], PORTS)

    def test_long_list_rejected(self):
        with pytest.raises(ValueError, match=r"5 entries but the model has 4"):
            resolve_port_mode_counts([1, 1, 1, 1, 1], PORTS)

    def test_unknown_port_name_rejected(self):
        with pytest.raises(ValueError, match=r"unknown port"):
            resolve_port_mode_counts({'portX': 2}, PORTS)

    def test_error_lists_the_valid_ports(self):
        with pytest.raises(ValueError) as e:
            resolve_port_mode_counts({'nope': 1}, PORTS)
        for p in PORTS:
            assert p in str(e.value)

    @pytest.mark.parametrize('spec', [0, [1, 0, 1, 1], {'port1': 0}])
    def test_counts_below_one_rejected(self, spec):
        with pytest.raises(ValueError, match=r">= 1"):
            resolve_port_mode_counts(spec, PORTS)


class TestPortMap:
    """fds.port_map() must work WITHOUT a solve, since its whole purpose is to
    tell the user what to put in the nportmodes list."""

    @staticmethod
    def _proj(tmp_path):
        p = EMProject(name='pmap', base_dir=str(tmp_path), overwrite=True)
        p.geometry = RectangularWaveguide(a=0.1, b=0.05, L=0.06667, maxh=0.06)
        return p

    def test_lists_ports_before_solving(self, tmp_path):
        rows = self._proj(tmp_path).fds.port_map()
        assert len(rows) >= 2
        assert [r['index'] for r in rows] == list(range(len(rows)))
        for r in rows:
            assert {'index', 'port', 'geometry', 'dims_mm', 'modes'} <= set(r)
            assert r['role'] in ('external', 'internal (join)')

    def test_order_matches_what_a_list_spec_indexes(self, tmp_path):
        fds = self._proj(tmp_path).fds
        names = [r['port'] for r in fds.port_map()]
        counts = resolve_port_mode_counts(list(range(1, len(names) + 1)), names)
        assert counts[names[0]] == 1
        assert counts[names[-1]] == len(names)

    def test_print_port_map_runs(self, tmp_path, capsys):
        self._proj(tmp_path).fds.print_port_map()
        out = capsys.readouterr().out
        assert 'nportmodes accepts' in out
        assert 'port1' in out


class TestEndToEnd:
    """A per-port spec must actually change the modes that get computed."""

    def test_list_spec_gives_per_port_mode_counts(self, tmp_path):
        p = EMProject(name='pm_e2e', base_dir=str(tmp_path), overwrite=True)
        p.geometry = RectangularWaveguide(a=0.1, b=0.05, L=0.06667, maxh=0.06)
        names = [r['port'] for r in p.fds.port_map()]
        spec = [2] + [1] * (len(names) - 1)
        p.fds.solve(config=dict(fmin=1.8, fmax=2.4, nsamples=2, order=1,
                                nportmodes=spec, solver_type='direct',
                                rerun=True, store_snapshots=False))
        pm = p.fds.port_solver.port_modes
        assert len(pm[names[0]]) == 2
        for n in names[1:]:
            assert len(pm[n]) == 1
        # Z is square in the TOTAL number of port-modes, not ports
        assert p.fds.fom._Z_matrix.shape[-1] == sum(spec)

    def test_bad_spec_raises_before_solving(self, tmp_path):
        p = EMProject(name='pm_bad', base_dir=str(tmp_path), overwrite=True)
        p.geometry = RectangularWaveguide(a=0.1, b=0.05, L=0.06667, maxh=0.06)
        with pytest.raises(ValueError):
            p.fds.solve(config=dict(fmin=1.8, fmax=2.4, nsamples=2, order=1,
                                    nportmodes=[1] * 99, solver_type='direct',
                                    rerun=True))


class TestAssemblies:
    """A GLUED assembly shares its joins, so the port count is not
    (ports per section) x (sections); a NETLIST never meshes the assembly at
    all and applies nportmodes per section."""

    A, B, L, MAXH = 0.1, 0.05, 0.06667, 0.06

    def _section(self):
        return RectangularWaveguide(a=self.A, b=self.B, L=self.L,
                                    maxh=self.MAXH)

    def _glued(self, tmp_path, name):
        p = EMProject(name=name, base_dir=str(tmp_path), overwrite=True)
        asm = p.create_assembly(main_axis='Z')
        sec = self._section()
        asm.add('s1', sec)
        asm.add('s2', sec, after='s1')
        asm.build()
        asm.generate_mesh(maxh=self.MAXH)
        p.geometry = asm
        return p

    def test_glued_assembly_shares_the_join(self, tmp_path):
        rows = self._glued(tmp_path, 'glue_pm').fds.port_map()
        # two 2-port sections glue to THREE ports, not four
        assert len(rows) == 3
        roles = [r['role'] for r in rows]
        assert roles.count('internal (join)') == 1
        assert roles.count('external') == 2

    def test_glued_assembly_takes_a_list_over_all_ports(self, tmp_path):
        names = [r['port'] for r in self._glued(tmp_path, 'glue_pm2').fds.port_map()]
        counts = resolve_port_mode_counts([1, 2, 1], names)
        assert counts[names[1]] == 2          # the join carries two modes
        with pytest.raises(ValueError):       # a per-section length is wrong here
            resolve_port_mode_counts([1, 1], names)

    def test_netlist_reports_ports_per_section(self, tmp_path):
        p = EMProject(name='net_pm', base_dir=str(tmp_path), overwrite=True)
        asm = p.create_assembly(main_axis='Z')
        asm.add('cell', self._section(), n=2)

        rows = p.fds.port_map()               # proj.fds has no mesh at all here
        assert rows, 'netlist port_map returned nothing'
        assert all('section' in r for r in rows)
        assert {r['section'] for r in rows} == {'cell'}
        # n=2 repeats ONE unique section, so the map lists that section once
        assert len(rows) == 2

    def test_netlist_print_explains_per_section_scope(self, tmp_path, capsys):
        p = EMProject(name='net_pm2', base_dir=str(tmp_path), overwrite=True)
        asm = p.create_assembly(main_axis='Z')
        asm.add('cell', self._section(), n=2)
        p.fds.print_port_map()
        out = capsys.readouterr().out
        assert 'PER SECTION' in out
        assert "section 'cell'" in out


class TestAssemblyChainGeometry:
    """An assembly has a geometry of the WHOLE chain but no mesh of its own."""

    A, B, L, MAXH = 0.1, 0.05, 0.06667, 0.06

    def _asm(self, tmp_path, n, name):
        p = EMProject(name=name, base_dir=str(tmp_path), overwrite=True)
        asm = p.create_assembly(main_axis='Z')
        asm.add('cell', RectangularWaveguide(a=self.A, b=self.B, L=self.L,
                                             maxh=self.MAXH), n=n)
        return p, asm

    def test_repeats_are_expanded_along_the_main_axis(self, tmp_path):
        _, asm1 = self._asm(tmp_path, 1, 'chain1')
        _, asm3 = self._asm(tmp_path, 3, 'chain3')
        z1 = asm1.build_chain_geometry().bounding_box
        z3 = asm3.build_chain_geometry().bounding_box
        len1 = z1[1][2] - z1[0][2]
        len3 = z3[1][2] - z3[0][2]
        assert len1 == pytest.approx(self.L, rel=1e-3)
        assert len3 == pytest.approx(3 * self.L, rel=1e-3)

    def test_chain_geometry_creates_no_mesh(self, tmp_path):
        _, asm = self._asm(tmp_path, 3, 'chain_nomesh')
        asm.build_chain_geometry()
        # sections carry their own meshes; the assembly must not
        assert getattr(asm, 'mesh', None) is None

    def test_geo_is_populated_for_a_netlist(self, tmp_path):
        p, asm = self._asm(tmp_path, 3, 'chain_geo')
        assert asm.geo is None            # never build()-ed on a netlist
        asm.build_chain_geometry()
        assert asm.geo is not None
        # and the netlist path is untouched by having drawn it
        assert p.fds._netlist_assembly() is not None
