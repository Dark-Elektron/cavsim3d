"""Rigid replication of a reference mesh into a compound (chain) mesh.

For state-space concatenation (SSC) of a chain of *identical* sections, the
full-order model is solved on ONE reference section and reused.  To visualise
the reconstructed field over the whole chain, each occurrence is a rigid copy
(translation, optional rotation) of the reference mesh placed at its location.

Because a rigid transform preserves mesh topology and DOF numbering exactly,
and an H(curl) edge/face/cell DOF value is invariant under a rigid transform
(``RᵀR = I``), the reconstructed coefficient vector of a section can be copied
*verbatim* onto its placed copy — no cross-mesh interpolation, no error.

This module provides:
  * :func:`replicate_mesh` — build a compound NGSolve mesh from N rigid
    placements of a reference mesh (disjoint blocks, each topologically equal
    to the reference).
  * :func:`block_dof_maps` — for a given FE space order, the per-block map
    ``reference DOF -> compound DOF`` (handles EDGE/FACE/CELL dofs, any order).
  * :func:`assemble_compound_field` — fill a compound ``GridFunction`` from
    per-section reference coefficient vectors.

Only the combinatorics + a coordinate transform are used, so the placement is
exact for translations and rotations (curved/high-order elements included).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from netgen.meshing import (
    Mesh as NgMesh, MeshPoint, Element3D, Element2D, FaceDescriptor,
)
from netgen.csg import Pnt as CPnt
from ngsolve import Mesh, GridFunction, NodeId, EDGE, FACE, CELL


@dataclass
class Placement:
    """A rigid placement of the reference section.

    translation : (tx, ty, tz)
    rotation    : optional (axis, angle_rad, center) about which to rotate
                  before translating; axis is a 3-vector, center a point.
    """
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Optional[Tuple[Sequence[float], float, Sequence[float]]] = None

    def matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (R, center, t) so that x' = R (x - center) + center + t."""
        t = np.asarray(self.translation, dtype=float)
        if self.rotation is None:
            return np.eye(3), np.zeros(3), t
        axis, angle, center = self.rotation
        a = np.asarray(axis, dtype=float)
        a = a / (np.linalg.norm(a) or 1.0)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R, np.asarray(center, dtype=float), t

    def apply(self, p: np.ndarray) -> np.ndarray:
        R, c, t = self.matrix()
        return R @ (p - c) + c + t


def replicate_mesh(ref_mesh: Mesh, placements: Sequence[Placement]) -> Mesh:
    """Build a compound mesh = N rigid copies of *ref_mesh*.

    Each copy is a disjoint block topologically identical to the reference, so
    per-block DOF numbering matches the reference (used by :func:`block_dof_maps`).
    Volume materials and boundary names are preserved, tagged per section as
    ``sec{i}/<name>`` so individual sections can still be selected/coloured.
    """
    specs = [(ref_mesh, p) for p in placements]
    mesh, _offsets = replicate_blocks(specs)
    return mesh


def replicate_blocks(specs):
    """Build a compound mesh from heterogeneous placed blocks.

    Parameters
    ----------
    specs : list of (mesh, Placement)
        Each entry places a (possibly different) reference *mesh* rigidly.
        Blocks are disjoint and each is topologically identical to its source
        mesh, so per-block DOF transfer is exact (see :func:`block_dof_maps`).

    Returns
    -------
    (compound_mesh, point_offsets) : (Mesh, list[int])
        ``point_offsets[b]`` is the global base point index of block *b*
        (needed to map that block's source DOFs onto the compound).
    """
    comp = NgMesh()
    comp.dim = 3

    point_offsets = []
    pt_off = 0
    dom_off = 0
    fd_off = 0

    for b, (src_mesh, place) in enumerate(specs):
        ng = src_mesh.ngmesh
        pts = ng.Points()
        npts = len(pts)
        mats = list(src_mesh.GetMaterials())
        bnds = list(src_mesh.GetBoundaries())
        n_fd = ng.GetNFaceDescriptors()

        point_offsets.append(pt_off)

        for i in range(1, npts + 1):
            p = np.asarray(pts[i].p, dtype=float)
            q = place.apply(p)
            comp.Add(MeshPoint(CPnt(float(q[0]), float(q[1]), float(q[2]))))

        for di, name in enumerate(mats, start=1):
            comp.SetMaterial(dom_off + di, f"sec{b}/{name}")

        for fi in range(1, n_fd + 1):
            fd = ng.FaceDescriptor(fi)
            bcname = bnds[fd.bc - 1] if 0 <= fd.bc - 1 < len(bnds) else f"bc{fd.bc}"
            comp.Add(FaceDescriptor(
                bc=fd_off + fi,
                domin=(fd.domin + dom_off) if fd.domin > 0 else 0,
                domout=(fd.domout + dom_off) if fd.domout > 0 else 0,
                surfnr=fd.surfnr,
            ))
            comp.SetBCName(fd_off + fi - 1, f"sec{b}/{bcname}")

        for el in ng.Elements3D():
            comp.Add(Element3D(el.index + dom_off, [v.nr + pt_off for v in el.vertices]))
        for el in ng.Elements2D():
            comp.Add(Element2D(el.index + fd_off, [v.nr + pt_off for v in el.vertices]))

        pt_off += npts
        dom_off += max(len(mats), 1)
        fd_off += max(n_fd, 1)

    return Mesh(comp), point_offsets


def _node_dof_map(ref_fes, comp_fes, ref_mesh, comp_mesh, base_pt: int):
    """Map reference DOFs to compound DOFs for one block.

    Builds correspondences for EDGE/FACE/CELL nodes by matching node vertex
    sets shifted by ``base_pt`` (the per-block point offset).  Returns
    ``(ref_dofs, comp_dofs)`` index arrays of equal length.
    """
    # Compound lookups: vertex-set -> node number, per node type.
    def comp_lookup(iterable, get_verts):
        d = {}
        for node in iterable:
            d[frozenset(get_verts(node))] = node
        return d

    comp_edges = comp_lookup(comp_mesh.edges, lambda e: [v.nr for v in e.vertices])
    comp_faces = comp_lookup(comp_mesh.faces, lambda f: [v.nr for v in f.vertices])
    comp_cells = comp_lookup(comp_mesh.Elements(), lambda c: [v.nr for v in c.vertices])

    ref_idx: List[int] = []
    comp_idx: List[int] = []

    def add(node_type, ref_nodes, comp_dict, ref_get_verts, comp_get_node):
        for node in ref_nodes:
            rdofs = list(ref_fes.GetDofNrs(NodeId(node_type, node.nr)))
            if not rdofs:
                continue
            key = frozenset(v + base_pt for v in ref_get_verts(node))
            cnode = comp_dict.get(key)
            if cnode is None:
                continue
            cdofs = list(comp_fes.GetDofNrs(NodeId(node_type, comp_get_node(cnode))))
            for rd, cd in zip(rdofs, cdofs):
                if rd >= 0 and cd >= 0:
                    ref_idx.append(rd)
                    comp_idx.append(cd)

    add(EDGE, ref_mesh.edges, comp_edges,
        lambda e: [v.nr for v in e.vertices], lambda n: n.nr)
    add(FACE, ref_mesh.faces, comp_faces,
        lambda f: [v.nr for v in f.vertices], lambda n: n.nr)
    add(CELL, ref_mesh.Elements(), comp_cells,
        lambda c: [v.nr for v in c.vertices], lambda n: n.nr)

    return np.array(ref_idx, dtype=np.int64), np.array(comp_idx, dtype=np.int64)


def block_dof_maps(ref_fes, comp_fes, ref_mesh, comp_mesh, n_blocks: int):
    """Per-block ``(ref_dofs, comp_dofs)`` maps for a *homogeneous* compound
    (all blocks share one reference mesh/FES)."""
    npts = len(ref_mesh.ngmesh.Points())
    return [
        _node_dof_map(ref_fes, comp_fes, ref_mesh, comp_mesh, b * npts)
        for b in range(n_blocks)
    ]


def hetero_block_dof_maps(block_fes_meshes, comp_fes, comp_mesh, point_offsets):
    """Per-block ``(ref_dofs, comp_dofs)`` maps for a *heterogeneous* compound.

    Parameters
    ----------
    block_fes_meshes : list of (ref_fes, ref_mesh)
        The source FE space + mesh for each block (may differ between blocks).
    comp_fes, comp_mesh : the compound FE space / mesh.
    point_offsets : list[int]
        Base point index of each block (from :func:`replicate_blocks`).
    """
    return [
        _node_dof_map(ref_fes, comp_fes, ref_mesh, comp_mesh, point_offsets[b])
        for b, (ref_fes, ref_mesh) in enumerate(block_fes_meshes)
    ]


def assemble_compound_field(comp_fes, section_vectors, dof_maps) -> GridFunction:
    """Fill a compound ``GridFunction`` from per-section reference vectors.

    Parameters
    ----------
    comp_fes : FESpace on the compound mesh.
    section_vectors : list of 1-D arrays, one reconstructed coefficient vector
        per section (in the *reference* FE space's DOF order).
    dof_maps : output of :func:`block_dof_maps` (same length / order).
    """
    gf = GridFunction(comp_fes)
    v = gf.vec.FV().NumPy()
    v[:] = 0.0
    for vec, (ref_idx, comp_idx) in zip(section_vectors, dof_maps):
        arr = np.asarray(vec).ravel()
        v[comp_idx] = arr[ref_idx]
    return gf
