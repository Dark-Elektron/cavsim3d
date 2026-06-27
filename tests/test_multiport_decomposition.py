"""Tests for multiport domain decomposition.

Covers the generalisation of the substructuring pipeline from linear
2-port cascades to arbitrary multiport domains:

  * ``OCCImporter.split`` labelling the cut regions as distinct sub-domains
    (``subdomain1/...``) connected through an internal split-plane port.
  * ``Assembly`` reusing the same component twice without corrupting
    materials (no double-prefix) or merging distinct ports (coax couplers
    at the same axial position stay separate).
  * ``FrequencyDomainSolver`` structure detection via mesh-derived
    port->domain adjacency (is_compound / internal / external /
    domain_port_map) for multiport domains.
  * ``ModelOrderReduction._build_connections`` building concatenation
    connections from shared interface ports for non-cascade topologies.
"""

import pytest
import numpy as np
from pathlib import Path

from cavsim3d.geometry.importers import OCCImporter
from cavsim3d.geometry.assembly import Assembly
from cavsim3d.solvers.frequency_domain import FrequencyDomainSolver
from cavsim3d.rom.reduction import ModelOrderReduction

NB = Path(__file__).parent.parent / "notebooks"
TEM = NB / "cavsim3d_tem_port_test.stp"
C3794 = NB / "c3794_4hc_1fpc_w_TEM.stp"

skip_no_tem = pytest.mark.skipif(not TEM.exists(), reason="cavsim3d_tem_port_test.stp missing")
skip_no_c3794 = pytest.mark.skipif(not C3794.exists(), reason="c3794_4hc_1fpc_w_TEM.stp missing")

TEM_MATS = {'solid5': 'PEC', 'solid2': {'eps_r': 10}, 'solid1': {'eps_r': 1}}
C3794_MATS = {'lh4*': 'PEC', 'dh2*': 'PEC', 'solid': 'PEC',
              'ceramic*': {'eps_r': 10}, 'solid1*': {'eps_r': 1}}


# ===========================================================================
# Unit: connection builder generalisation (no mesh needed)
# ===========================================================================

class TestConnectionBuilder:
    """ModelOrderReduction._build_connections for arbitrary topologies."""

    @staticmethod
    def _rom(domains, internal, adjacency, domain_port_map):
        rom = object.__new__(ModelOrderReduction)
        rom.domains = domains
        rom.n_domains = len(domains)
        rom._internal_ports = internal
        rom._port_domain_adjacency = adjacency
        rom.domain_port_map = domain_port_map
        return rom

    def test_multiport_chain_connections(self):
        """Three domains, the middle one carrying an extra (coax) port."""
        rom = self._rom(
            domains=['d1', 'd2', 'd3'],
            internal=['pA', 'pB'],
            adjacency={
                'p_end1': {'d1'}, 'pA': {'d1', 'd2'},
                'p_coax': {'d2'}, 'pB': {'d2', 'd3'}, 'p_end3': {'d3'},
            },
            domain_port_map={
                'd1': ['p_end1', 'pA'],
                'd2': ['pA', 'p_coax', 'pB'],
                'd3': ['pB', 'p_end3'],
            },
        )
        conns = rom._build_connections()
        assert ((0, 'pA'), (1, 'pA')) in conns
        assert ((1, 'pB'), (2, 'pB')) in conns
        assert len(conns) == 2

    def test_single_interface_two_domains(self):
        rom = self._rom(
            domains=['a', 'b'],
            internal=['junction'],
            adjacency={'pa': {'a'}, 'junction': {'a', 'b'}, 'pb': {'b'}},
            domain_port_map={'a': ['pa', 'junction'], 'b': ['junction', 'pb']},
        )
        conns = rom._build_connections()
        assert conns == [((0, 'junction'), (1, 'junction'))]

    def test_legacy_fallback_without_adjacency(self):
        """Without adjacency data, fall back to the sequential chain."""
        rom = self._rom(
            domains=['d1', 'd2'],
            internal=[],
            adjacency={},
            domain_port_map={'d1': ['p1', 'p2'], 'd2': ['p2', 'p3']},
        )
        conns = rom._build_connections()
        assert conns == [((0, 'p2'), (1, 'p2'))]


# ===========================================================================
# Geometry: split -> sub-domains
# ===========================================================================

@skip_no_tem
class TestSplitSubdomains:

    @pytest.fixture(scope="class")
    def split_geo(self):
        geo = OCCImporter(str(TEM), unit='mm', auto_build=False)
        geo.set_materials(TEM_MATS)
        pmin, pmax = geo.get_bounding_box()
        geo.add_splitting_plane_at_y((pmin[1] + pmax[1]) / 2)
        geo.split()
        geo.generate_mesh(maxh=0.05)
        return geo

    def test_two_subdomains_declared(self, split_geo):
        assert split_geo.domains == ['subdomain1', 'subdomain2']

    def test_internal_port_created(self, split_geo):
        assert len(split_geo.internal_ports) >= 1

    def test_materials_prefixed_in_mesh(self, split_geo):
        mats = set(split_geo.mesh.GetMaterials())
        assert any(m.startswith('subdomain1/') for m in mats)
        assert any(m.startswith('subdomain2/') for m in mats)

    def test_dielectric_preserved_through_prefix(self, split_geo):
        """eps_r must survive the region prefixing (1 vacuum, 10 ceramic)."""
        eps_values = {
            split_geo.get_material(m)['eps_r']
            for m in set(split_geo.mesh.GetMaterials())
        }
        assert eps_values <= {1.0, 10.0}
        assert 10.0 in eps_values  # ceramic still present and correct

    def test_internal_port_is_a_mesh_boundary(self, split_geo):
        bnds = set(split_geo.mesh.GetBoundaries())
        for p in split_geo.internal_ports:
            assert p in bnds


# ===========================================================================
# Solver: structure detection for a split single geometry
# ===========================================================================

@skip_no_tem
class TestSplitSolverStructure:

    @pytest.fixture(scope="class")
    def fds(self):
        geo = OCCImporter(str(TEM), unit='mm', auto_build=False)
        geo.set_materials(TEM_MATS)
        pmin, pmax = geo.get_bounding_box()
        geo.add_splitting_plane_at_y((pmin[1] + pmax[1]) / 2)
        geo.split()
        geo.generate_mesh(maxh=0.05)
        solver = FrequencyDomainSolver(geo, order=1)
        solver.mesh = geo.mesh
        return solver

    def test_is_compound(self, fds):
        assert fds.is_compound is True

    def test_domains_are_subdomains(self, fds):
        assert set(fds.domains) == {'subdomain1', 'subdomain2'}

    def test_internal_ports_shared_by_two_domains(self, fds):
        assert len(fds.internal_ports) >= 1
        for p in fds.internal_ports:
            assert len(fds._port_domain_adjacency[p]) == 2

    def test_external_ports_touch_single_domain(self, fds):
        for p in fds.external_ports:
            assert len(fds._port_domain_adjacency.get(p, set())) == 1

    def test_interface_port_in_both_domain_maps(self, fds):
        for p in fds.internal_ports:
            owners = [d for d in fds.domains if p in fds.domain_port_map[d]]
            assert len(owners) == 2


# ===========================================================================
# Assembly: identical components + multiport structure (c3794, gated)
# ===========================================================================

@skip_no_c3794
class TestAssemblyMultiport:

    @pytest.fixture(scope="class")
    def asm_and_source(self):
        g1 = OCCImporter(str(C3794), unit='mm', auto_build=False)
        g1.set_materials(C3794_MATS)
        asm = Assembly(main_axis='Z')
        asm.add('geo1', g1)
        asm.add('geo2', g1, after='geo1')
        asm.generate_mesh(maxh=5, curvaturesafety=1.5)
        return asm, g1

    def test_no_material_double_prefix(self, asm_and_source):
        asm, _ = asm_and_source
        for mats in asm._domain_materials.values():
            for m in mats:
                assert m.count('/') == 1, f"double-prefixed material: {m}"
                assert not m.startswith('geo2/geo1')

    def test_source_geometry_left_pristine(self, asm_and_source):
        """Reusing the same object must not mutate its materials."""
        _, g1 = asm_and_source
        assert all('/' not in s.name for s in g1.geo.solids)

    def test_dielectric_per_domain(self, asm_and_source):
        asm, _ = asm_and_source
        for m in set(asm.mesh.GetMaterials()):
            assert asm.get_material(m)['eps_r'] in (1.0, 10.0)

    def test_coax_ports_not_merged(self, asm_and_source):
        """Two coax couplers at the same axial position stay distinct ports."""
        asm, _ = asm_and_source
        # Each single cavity has 4 ports; two cavities sharing one junction
        # => 7 distinct ports.  A position-only grouping would collapse the
        # same-Z coax pairs and yield fewer.
        ports = [b for b in set(asm.mesh.GetBoundaries()) if b.startswith('port')]
        assert len(ports) == 7

    def test_solver_multiport_structure(self, asm_and_source):
        asm, _ = asm_and_source
        fds = FrequencyDomainSolver(asm, order=1)
        fds.mesh = asm.mesh
        assert fds.is_compound
        assert len(fds.domains) == 2
        # exactly one shared junction
        assert len(fds.internal_ports) == 1
        # at least one domain has MORE than two ports (true multiport)
        assert any(len(ps) > 2 for ps in fds.domain_port_map.values())
        # the interface port is shared by both domains
        ip = fds.internal_ports[0]
        assert all(ip in fds.domain_port_map[d] for d in fds.domains)


# ===========================================================================
# Variable number of modes per port
# ===========================================================================

class TestVariablePortModesUnit:
    """ReducedStructure / connection handling with non-uniform modes per port."""

    def test_reduced_structure_port_mode_pairs(self):
        from cavsim3d.rom.structures import ReducedStructure
        # port1: 2 modes, port2: 1 mode  -> 3 columns in Brd
        rs = ReducedStructure(
            Ard=np.eye(4),
            Brd=np.zeros((4, 3)),
            ports=['port1', 'port2'],
            port_modes={'port1': {0: None, 1: None}, 'port2': {0: None}},
            domain='d',
        )
        assert rs.port_mode_pairs == [('port1', 0), ('port1', 1), ('port2', 0)]
        assert rs.n_total_port_modes == 3
        assert rs.get_port_mode_column('port2', 0) == 2

    def test_reduced_structure_rejects_wrong_columns(self):
        from cavsim3d.rom.structures import ReducedStructure
        with pytest.raises(ValueError):
            ReducedStructure(
                Ard=np.eye(2), Brd=np.zeros((2, 5)),  # wrong: should be 3
                ports=['port1', 'port2'],
                port_modes={'port1': {0: None, 1: None}, 'port2': {0: None}},
            )


class TestVariablePortModesIntegration:
    """Full coupled + concatenation chain with different modes per port."""

    A, L, MAXH = 0.1, 0.1, 0.04

    def _assembly(self):
        from cavsim3d.geometry.primitives import RectangularWaveguide
        asm = Assembly(main_axis='Z')
        asm.add('h1', RectangularWaveguide(a=self.A, L=self.L, maxh=self.MAXH))
        asm.add('h2', RectangularWaveguide(a=self.A, L=self.L, maxh=self.MAXH), after='h1')
        asm.generate_mesh(maxh=self.MAXH)
        return asm

    def test_concatenation_with_variable_modes(self):
        asm = self._assembly()
        solver = FrequencyDomainSolver(asm, order=2)
        # junction port2 must match on both sides (2); externals differ (2 vs 1)
        solver.solve(fmin=1.8, fmax=2.4, nsamples=3, per_domain=True,
                     store_snapshots=True, global_method=None,
                     nportmodes={'port1': 2, 'port2': 2, 'port3': 1})

        assert {p: len(m) for p, m in solver.port_modes.items()} == \
            {'port1': 2, 'port2': 2, 'port3': 1}

        rom = ModelOrderReduction(solver)
        rom.reduce(tol=1e-9)
        concat = rom.concatenate()

        # External port-modes = port1(2) + port3(1) = 3; junction eliminated.
        assert len(concat.ports) == 3
        res = concat.solve(fmin=1.8, fmax=2.4, nsamples=5)
        assert res['Z'].shape == (5, 3, 3)
        assert np.all(np.isfinite(res['Z']))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
