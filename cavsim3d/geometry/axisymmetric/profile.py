"""Meridian profiles of axisymmetric structures, and their 3D revolution.

A :class:`Profile` is the *blueprint* of a body of revolution: an ordered,
closed list of boundary segments (straight lines, circular arcs, exact ellipse
arcs, splines) in the (z, r) meridian plane, each tagged with a boundary name:

``'AXI'``
    The symmetry axis. It closes the contour and vanishes on revolution.
``'PEC'``
    A conducting wall.
``'PMC'``
    A beam aperture: a flat end face from the axis out to the bore. It
    becomes a port face (``port1``, ``port2``, ... in increasing z).

:func:`revolve` sweeps a profile 360 degrees about the beam axis (Z) into a
named, meshable netgen.occ solid. The profile code is ported from cavsim2d, so
a contour defined there is the same contour here. Coordinates are in metres.

Example (a pillbox with beam apertures)::

    p = (Profile('pillbox')
         .start(-0.05, 0.0)
         .line_to(-0.05, 0.02, 'PMC')    # left aperture
         .line_to(-0.05, 0.10, 'PEC')    # left end plate
         .line_to(0.05, 0.10, 'PEC')     # barrel
         .line_to(0.05, 0.02, 'PEC')     # right end plate
         .line_to(0.05, 0.0, 'PMC')      # right aperture
         .close('AXI'))
    solid = revolve(p)
"""
import warnings

import numpy as np
from scipy.interpolate import BSpline
from scipy.special import comb


class MaterialRegion:
    """An axisymmetric sub-domain of a :class:`Profile` with its own material.

    ``z=(z0, z1), r=(r0, r1)`` gives a rectangular ring (an annular cylinder in
    3D). ``points=[(z, r), ...]`` gives an arbitrary closed polygon, which
    revolves into a solid of revolution: a triangle becomes a cone. Coordinates
    are in metres. The region is clipped to the profile it is added to, so it
    may be specified generously.
    """

    def __init__(self, material, z=None, r=None, points=None):
        self.material = str(material)
        if points is not None:
            if z is not None or r is not None:
                raise ValueError(
                    f"region {self.material!r}: give either 'points' or 'z'/'r', not both.")
            pts = [(float(a), float(b)) for a, b in points]
            if len(pts) > 1 and pts[0] == pts[-1]:
                pts = pts[:-1]
            if len(pts) < 3:
                raise ValueError(
                    f"region {self.material!r}: 'points' needs at least 3 distinct "
                    f"(z, r) vertices, got {len(pts)}.")
            if any(b < 0 for _, b in pts):
                raise ValueError(
                    f"region {self.material!r} has a vertex at r < 0; the meridian "
                    "plane is r >= 0.")
            n = len(pts)
            area2 = sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
                        for i in range(n))
            if abs(area2) < 1e-30:
                raise ValueError(f"region {self.material!r}: 'points' encloses zero area.")
            # Counter-clockwise, so the boolean pieces glue conformally.
            self.points = pts if area2 > 0 else pts[::-1]
            self._is_rect = False
            self.z = (min(a for a, _ in pts), max(a for a, _ in pts))
            self.r = (min(b for _, b in pts), max(b for _, b in pts))
            return
        if z is None or r is None:
            raise ValueError(f"region {self.material!r}: give 'z' and 'r', or 'points'.")
        self.z = (float(min(z)), float(max(z)))
        self.r = (float(min(r)), float(max(r)))
        if self.z[0] == self.z[1] or self.r[0] == self.r[1]:
            raise ValueError(
                f"region {self.material!r} has zero extent (z={self.z}, r={self.r}).")
        if self.r[0] < 0:
            raise ValueError(f"region {self.material!r} has r < 0 ({self.r}).")
        self._is_rect = True
        self.points = [(self.z[0], self.r[0]), (self.z[1], self.r[0]),
                       (self.z[1], self.r[1]), (self.z[0], self.r[1])]

    def __repr__(self):
        if self._is_rect:
            return f"MaterialRegion({self.material!r}, z={self.z}, r={self.r})"
        return f"MaterialRegion({self.material!r}, points={self.points})"

    def translated(self, dz):
        """A copy of this region shifted by *dz* along z."""
        if self._is_rect:
            return MaterialRegion(self.material, z=(self.z[0] + dz, self.z[1] + dz), r=self.r)
        return MaterialRegion(self.material, points=[(z + dz, r) for z, r in self.points])

    def to_occ_face(self):
        """A netgen.occ face covering the region (before clipping to the profile)."""
        from netgen.occ import WorkPlane
        if self._is_rect:
            return (WorkPlane().MoveTo(self.z[0], self.r[0])
                    .Rectangle(self.z[1] - self.z[0], self.r[1] - self.r[0]).Face())
        wp = WorkPlane().MoveTo(*self.points[0])
        for zv, rv in self.points[1:]:
            wp = wp.LineTo(zv, rv)
        return wp.Close().Face()


class Profile:
    """An ordered, closed meridian contour in the (z, r) plane, in metres."""

    #: Material name of whatever is not claimed by a :class:`MaterialRegion`.
    default_material = 'vacuum'

    def __init__(self, name='profile'):
        self.name = name
        self._pts = []      # ordered boundary points [(z, r), ...]
        self._segs = []     # [{'kind', 'i0', 'i1', 'name', ...}, ...]
        self._regions = []  # [MaterialRegion, ...]

    # -- construction -------------------------------------------------------

    def start(self, z, r):
        """Set the starting point of the contour."""
        self._pts = [(float(z), float(r))]
        self._segs = []
        return self

    def line_to(self, z, r, boundary):
        """Straight segment from the current point to (z, r), tagged *boundary*."""
        i0 = len(self._pts) - 1
        self._pts.append((float(z), float(r)))
        self._segs.append({'kind': 'line', 'i0': i0, 'i1': i0 + 1, 'name': boundary})
        return self

    def arc_to(self, z, r, through, boundary):
        """Circular arc from the current point to (z, r) through ``through=(z_m, r_m)``."""
        i0 = len(self._pts) - 1
        self._pts.append((float(z), float(r)))
        self._segs.append({'kind': 'arc', 'i0': i0, 'i1': i0 + 1, 'name': boundary,
                           'mid': (float(through[0]), float(through[1]))})
        return self

    def circle_arc_to(self, z, r, center, boundary):
        """Circular arc from the current point to (z, r) about ``center``.

        The short sweep is taken; the midpoint is placed on the circle and the
        segment is built as an exact three-point arc.
        """
        p0 = self._pts[-1]
        cz, cr = float(center[0]), float(center[1])
        radius = np.hypot(p0[0] - cz, p0[1] - cr)
        a0 = np.arctan2(p0[1] - cr, p0[0] - cz)
        a1 = np.arctan2(r - cr, z - cz)
        dt = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
        am = a0 + dt / 2.0
        mid = (cz + radius * np.cos(am), cr + radius * np.sin(am))
        return self.arc_to(z, r, through=mid, boundary=boundary)

    def ellipse_arc_to(self, z, r, center, semi_z, semi_r, boundary):
        """Exact elliptical arc from the current point to (z, r).

        The arc lies on the ellipse centred at ``center=(zc, rc)`` with
        semi-axis ``semi_z`` along z and ``semi_r`` along r; the short sweep is
        taken. Built as an exact OCC conic, not a polyline.
        """
        i0 = len(self._pts) - 1
        self._pts.append((float(z), float(r)))
        self._segs.append({'kind': 'ellipse', 'i0': i0, 'i1': i0 + 1, 'name': boundary,
                           'center': (float(center[0]), float(center[1])),
                           'semi_z': float(semi_z), 'semi_r': float(semi_r)})
        return self

    def spline_to(self, poles, boundary, kind='bspline', degree=3):
        """Free-form spline from the current point through the control ``poles``.

        The current point is the first pole and the last pole is the end point;
        both curve types are clamped, so the curve starts and ends on the
        contour. ``kind`` is ``'bspline'`` or ``'bezier'``.
        """
        kind = kind.lower()
        if kind not in ('bspline', 'bezier'):
            raise ValueError(f"spline kind must be 'bspline' or 'bezier', got {kind!r}")
        poles = [(float(z), float(r)) for z, r in poles]
        if len(poles) < 2:
            raise ValueError('a spline segment needs at least two poles')
        i0 = len(self._pts) - 1
        self._pts.append(poles[-1])
        self._segs.append({'kind': 'spline', 'i0': i0, 'i1': i0 + 1, 'name': boundary,
                           'interior': poles[:-1], 'spline_kind': kind,
                           'degree': int(degree)})
        return self

    def close(self, boundary):
        """Straight segment back to the start, tagged *boundary* (the axis)."""
        i0 = len(self._pts) - 1
        self._segs.append({'kind': 'line', 'i0': i0, 'i1': 0, 'name': boundary})
        return self

    def add_region(self, material, *, z=None, r=None, points=None):
        """Add a material sub-domain (a rectangular ring or a polygon), in metres.

        Regions are clipped in order, so where two overlap the first one wins.
        """
        if material == self.default_material:
            raise ValueError(f"region name {material!r} is the background material.")
        if any(reg.material == material for reg in self._regions):
            raise ValueError(f"a region named {material!r} was already added.")
        self._regions.append(MaterialRegion(material, z, r, points=points))
        return self

    def regions(self):
        """The material regions, in the order they were added."""
        return list(self._regions)

    # -- chaining -----------------------------------------------------------

    def _replay_segment(self, dst, seg, pts, dz):
        """Re-emit *seg* (of this profile) onto *dst*, translated by ``dz`` in z."""
        z1, r1 = pts[seg['i1']]
        z1 += dz
        k, name = seg['kind'], seg['name']
        if k == 'line':
            dst.line_to(z1, r1, name)
        elif k == 'arc':
            mz, mr = seg['mid']
            dst.arc_to(z1, r1, through=(mz + dz, mr), boundary=name)
        elif k == 'ellipse':
            cz, cr = seg['center']
            dst.ellipse_arc_to(z1, r1, center=(cz + dz, cr),
                               semi_z=seg['semi_z'], semi_r=seg['semi_r'], boundary=name)
        elif k == 'spline':
            poles = [(pz + dz, pr) for pz, pr in seg['interior']] + [(z1, r1)]
            dst.spline_to(poles, name, kind=seg['spline_kind'], degree=seg['degree'])
        else:
            raise ValueError(f'cannot chain segment kind {k!r}')

    def chained(self, n, spacing=None):
        """A new profile of *n* copies of this one chained end-to-end.

        The internal apertures are dropped and adjacent beam pipes merge into
        the inter-cavity drift, so the module is one connected vacuum region
        with a single aperture at each end.

        Parameters
        ----------
        n : int
            Number of copies. ``n <= 1`` returns ``self``.
        spacing : float or sequence of float, optional
            Inter-cavity drift length(s), iris to iris, in **metres**. When
            given, the internal beam-pipe stubs are dropped and each gap is set
            to exactly this length (one value for every gap, or ``n - 1``
            values). ``None`` keeps both stubs, so the drift is the base
            profile's two beam pipes.
        """
        n = int(n)
        if n <= 1:
            return self
        if self._regions:
            raise NotImplementedError('chaining a profile with material regions is not supported.')
        pts, segs = self._pts, self._segs
        if len(segs) < 3:
            raise ValueError('profile too simple to chain (need aperture/wall/aperture).')
        left_ap, right_ap, axi = segs[0], segs[-2], segs[-1]
        wall = segs[1:-2]
        if abs(pts[left_ap['i0']][1]) > 1e-9 or abs(pts[right_ap['i1']][1]) > 1e-9:
            raise ValueError('chained() needs the standard axis-to-axis meridian.')
        z_start = pts[left_ap['i1']][0]
        z_end = pts[right_ap['i0']][0]
        Ri = pts[left_ap['i1']][1]
        Ri_r = pts[right_ap['i0']][1]

        # Split the wall into [left stub | core | right stub]: a stub is the
        # leading/trailing run of horizontal lines at the beam-pipe radius. A
        # bare tube (no body above the pipe radius) keeps its whole wall as core.
        wall_r = [pts[s['i0']][1] for s in wall] + [pts[s['i1']][1] for s in wall]
        strip = wall and (max(wall_r) - max(Ri, Ri_r) > 1e-9)

        def _hstub(seg, r):
            return (seg['kind'] == 'line'
                    and abs(pts[seg['i0']][1] - r) < 1e-9
                    and abs(pts[seg['i1']][1] - r) < 1e-9)

        lo, hi = 0, len(wall)
        if strip:
            while lo < hi and _hstub(wall[lo], Ri):
                lo += 1
            while hi > lo and _hstub(wall[hi - 1], Ri_r):
                hi -= 1
        core = wall[lo:hi] or wall
        if core is wall:
            lo, hi = 0, len(wall)
        z_core0 = pts[core[0]['i0']][0]
        z_core1 = pts[core[-1]['i1']][0]
        core_len = z_core1 - z_core0
        left_stub = z_core0 - z_start
        right_stub = z_end - z_core1

        if spacing is None:
            gaps = [left_stub + right_stub] * (n - 1)
        else:
            try:
                gaps = [float(g) for g in spacing]
            except TypeError:
                gaps = [float(spacing)]
            if len(gaps) == 1:
                gaps = gaps * (n - 1)
            if len(gaps) != n - 1:
                raise ValueError(
                    'spacing must have one entry per inter-cavity gap: its length '
                    f'must equal chain - 1 == {n - 1}, but got {len(gaps)}.')
            if any(g < 0 for g in gaps):
                raise ValueError('spacing must be non-negative.')

        stub_name = wall[0]['name'] if lo > 0 else core[0]['name']
        p = Profile(self.name)
        p.start(pts[left_ap['i0']][0], 0.0)
        p.line_to(z_start, Ri, left_ap['name'])
        cursor = z_start
        if left_stub > 1e-12:
            cursor = z_core0
            p.line_to(cursor, Ri, stub_name)
        for c in range(n):
            dz = cursor - z_core0
            for s in core:
                self._replay_segment(p, s, pts, dz)
            cursor += core_len
            if c < n - 1 and gaps[c] > 1e-12:
                cursor += gaps[c]
                p.line_to(cursor, Ri, stub_name)
            elif c < n - 1:
                cursor += gaps[c]
        if right_stub > 1e-12:
            cursor += right_stub
            p.line_to(cursor, Ri_r, wall[-1]['name'] if hi < len(wall) else stub_name)
        p.line_to(cursor, 0.0, right_ap['name'])
        p.close(axi['name'])
        return p

    # -- spline helpers -----------------------------------------------------

    @classmethod
    def _spline_poles(cls, seg, pts):
        """Full control polygon: start point, interior poles, end point."""
        return [pts[seg['i0']]] + list(seg['interior']) + [pts[seg['i1']]]

    @staticmethod
    def _clamped_uniform_knots(n_poles, degree):
        """Knot vector of a clamped uniform B-spline (what OCC and gmsh use)."""
        n_internal = n_poles - degree - 1
        internal = list(np.arange(1, n_internal + 1) / (n_internal + 1)) if n_internal > 0 else []
        return np.array([0.0] * (degree + 1) + internal + [1.0] * (degree + 1))

    @classmethod
    def _spline_degree(cls, seg, pts):
        return min(seg['degree'], len(cls._spline_poles(seg, pts)) - 1)

    @staticmethod
    def _insert_knot(knots, poles, degree, u):
        """Boehm's algorithm: insert knot *u* once, leaving the curve unchanged."""
        k = int(np.searchsorted(knots, u, side='right')) - 1
        new = list(poles[:k - degree + 1])
        for i in range(k - degree + 1, k + 1):
            denom = knots[i + degree] - knots[i]
            a = 0.0 if denom == 0 else (u - knots[i]) / denom
            new.append((1.0 - a) * poles[i - 1] + a * poles[i])
        new.extend(poles[k:])
        return np.insert(knots, k + 1, u), np.array(new)

    @classmethod
    def _bspline_to_bezier(cls, poles, degree):
        """Split a clamped uniform B-spline into its exact Bezier segments.

        netgen's ``BSplineCurve`` is unclamped, so it does not end on the outer
        poles. Raising every internal knot to multiplicity ``degree`` decomposes
        the same curve into Bezier arcs, which netgen represents exactly.
        """
        poles = np.asarray(poles, dtype=float)
        knots = cls._clamped_uniform_knots(len(poles), degree)
        for u in sorted({float(k) for k in knots if 0.0 < k < 1.0}):
            while int(np.sum(np.isclose(knots, u))) < degree:
                knots, poles = cls._insert_knot(knots, poles, degree, u)
        n_seg = (len(poles) - 1) // degree
        return [poles[i * degree: i * degree + degree + 1] for i in range(n_seg)]

    @classmethod
    def _spline_points(cls, seg, pts, n=48):
        """Sample points along a spline segment."""
        poles = np.asarray(cls._spline_poles(seg, pts), dtype=float)
        u = np.linspace(0.0, 1.0, n)
        if seg['spline_kind'] == 'bezier':
            m = len(poles) - 1
            k = np.arange(m + 1)
            basis = comb(m, k)[None, :] * (u[:, None] ** k[None, :]) * ((1 - u)[:, None] ** (m - k)[None, :])
            return [tuple(p) for p in basis @ poles]
        deg = cls._spline_degree(seg, pts)
        knots = cls._clamped_uniform_knots(len(poles), deg)
        return [tuple(p) for p in BSpline(knots, poles, deg)(u)]

    @classmethod
    def _spline_speed(cls, seg, pts, u):
        """|dC/du| of a spline segment at parameters ``u``."""
        poles = np.asarray(cls._spline_poles(seg, pts), dtype=float)
        if seg['spline_kind'] == 'bezier':
            m = len(poles) - 1
            dpoles = m * np.diff(poles, axis=0)
            k = np.arange(m)
            basis = comb(m - 1, k)[None, :] * (u[:, None] ** k[None, :]) \
                * ((1 - u)[:, None] ** (m - 1 - k)[None, :])
            d = basis @ dpoles
        else:
            deg = cls._spline_degree(seg, pts)
            knots = cls._clamped_uniform_knots(len(poles), deg)
            d = BSpline(knots, poles, deg).derivative()(u)
        return np.linalg.norm(d, axis=1)

    def stationary_corners(self, rtol=1e-6):
        """Interior spline points where the tangent vanishes (``|dC/du| = 0``).

        A control polygon that reverses on itself (e.g. a multicell B-spline
        built by repeating one cell's polygon) has one at the iris. netgen
        cannot mesh the revolved surface there, so :func:`revolve` refuses it.
        """
        corners = []
        for s in self._segs:
            if s['kind'] != 'spline':
                continue
            u = np.linspace(0.0, 1.0, 1025)
            speed = self._spline_speed(s, self._pts, u)
            scale = speed.max()
            if scale <= 0:
                continue
            interior = np.flatnonzero(speed[1:-1] < rtol * scale) + 1
            pts = np.asarray(self._spline_points(s, self._pts, n=1025))
            for i in interior:
                p = tuple(pts[i])
                if not any(np.hypot(p[0] - c[0], p[1] - c[1]) < 1e-9 for c in corners):
                    corners.append(p)
        return corners

    # -- ellipse helpers ----------------------------------------------------

    @staticmethod
    def _ellipse_frame(semi_z, semi_r):
        """(major, minor, xdir) with major >= minor; xdir is the major axis."""
        if semi_z >= semi_r:
            return semi_z, semi_r, (1.0, 0.0)
        return semi_r, semi_z, (0.0, 1.0)

    @classmethod
    def _ellipse_param(cls, p, center, major, minor, xdir):
        """Parameter t with P = C + major*cos(t)*xdir + minor*sin(t)*ydir."""
        ux, uy = xdir
        vx, vy = -uy, ux
        dz, dr = p[0] - center[0], p[1] - center[1]
        du = (dz * ux + dr * uy) / major
        dv = (dz * vx + dr * vy) / minor
        return np.arctan2(dv, du)

    @classmethod
    def _ellipse_span(cls, seg, pts):
        """(center, major, minor, xdir, t_lo, t_hi), taking the short sweep."""
        c = seg['center']
        major, minor, xdir = cls._ellipse_frame(seg['semi_z'], seg['semi_r'])
        t0 = cls._ellipse_param(pts[seg['i0']], c, major, minor, xdir)
        t1 = cls._ellipse_param(pts[seg['i1']], c, major, minor, xdir)
        dt = t1 - t0
        while dt <= -np.pi:
            dt += 2 * np.pi
        while dt > np.pi:
            dt -= 2 * np.pi
        t_lo, t_hi = (t0, t0 + dt) if dt >= 0 else (t0 + dt, t0)
        return c, major, minor, xdir, t_lo, t_hi

    @classmethod
    def _ellipse_points(cls, seg, pts, n=24):
        """Sample points along an ellipse segment, ordered from ``i0`` to ``i1``."""
        c, major, minor, xdir, t_lo, t_hi = cls._ellipse_span(seg, pts)
        ux, uy = xdir
        vx, vy = -uy, ux
        out = []
        for t in np.linspace(t_lo, t_hi, n):
            ct, st = np.cos(t), np.sin(t)
            out.append((c[0] + major * ct * ux + minor * st * vx,
                        c[1] + major * ct * uy + minor * st * vy))
        p0 = pts[seg['i0']]
        if (np.hypot(out[0][0] - p0[0], out[0][1] - p0[1])
                > np.hypot(out[-1][0] - p0[0], out[-1][1] - p0[1])):
            out.reverse()
        return out

    def _arc_points(self, seg, n):
        """Sample a three-point circular arc segment."""
        p0 = np.asarray(self._pts[seg['i0']], dtype=float)
        p1 = np.asarray(self._pts[seg['i1']], dtype=float)
        pm = np.asarray(seg['mid'], dtype=float)
        ax, ay = p0
        bx, by = pm
        cx, cy = p1
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-18:
            return [tuple(p0), tuple(p1)]
        ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay)
              + (cx ** 2 + cy ** 2) * (ay - by)) / d
        uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx)
              + (cx ** 2 + cy ** 2) * (bx - ax)) / d
        radius = np.hypot(ax - ux, ay - uy)
        a0 = np.arctan2(ay - uy, ax - ux)
        am = np.arctan2(by - uy, bx - ux)
        a1 = np.arctan2(cy - uy, cx - ux)

        def unwrap(a, ref):
            while a - ref > np.pi:
                a -= 2 * np.pi
            while a - ref < -np.pi:
                a += 2 * np.pi
            return a

        am = unwrap(am, a0)
        a1 = unwrap(a1, am)
        return [(ux + radius * np.cos(t), uy + radius * np.sin(t))
                for t in np.linspace(a0, a1, n)]

    # -- queries ------------------------------------------------------------

    def segment_points(self, seg, n=24):
        """Sample *n* points along one segment, endpoints included."""
        if seg['kind'] == 'line':
            p0 = np.asarray(self._pts[seg['i0']], dtype=float)
            p1 = np.asarray(self._pts[seg['i1']], dtype=float)
            return [tuple(p0 + (p1 - p0) * t) for t in np.linspace(0.0, 1.0, max(2, n))]
        if seg['kind'] == 'arc':
            return self._arc_points(seg, max(3, n))
        if seg['kind'] == 'ellipse':
            return self._ellipse_points(seg, self._pts, n=max(3, n))
        if seg['kind'] == 'spline':
            return self._spline_points(seg, self._pts, n=max(3, n))
        raise ValueError(f"unknown segment kind {seg['kind']!r}")

    def contour_points(self, n=48):
        """The closed contour as an ordered ``(N, 2)`` array of ``(z, r)`` points.

        Curved segments are sampled with *n* points each; straight segments keep
        their two endpoints. Useful to plot the meridian, or to compute its area
        and centroid.
        """
        out = []
        for seg in self._segs:
            pts = self.segment_points(seg, 2 if seg['kind'] == 'line' else n)
            if out and np.allclose(out[-1], pts[0], atol=1e-12):
                pts = pts[1:]
            out.extend(tuple(map(float, p)) for p in pts)
        if len(out) > 1 and np.allclose(out[0], out[-1], atol=1e-12):
            out = out[:-1]
        return np.asarray(out, dtype=float)

    @property
    def points(self):
        """Ordered segment end points ``[(z, r), ...]`` (the start is not repeated)."""
        return list(self._pts)

    @property
    def segments(self):
        """The segment records, in contour order."""
        return list(self._segs)

    def apertures(self):
        """``[(z, r_bore), ...]`` of every ``'PMC'`` aperture, in increasing z.

        An aperture is a flat end face from the axis out to the bore, so it
        must be a straight line at constant z with one end on the axis.
        """
        out = []
        for s in self._segs:
            if s['name'] != 'PMC':
                continue
            (z0, r0), (z1, r1) = self._pts[s['i0']], self._pts[s['i1']]
            if s['kind'] != 'line' or abs(z1 - z0) > 1e-12 or min(abs(r0), abs(r1)) > 1e-12:
                raise ValueError(
                    f"profile {self.name!r}: a 'PMC' segment must be a flat aperture "
                    f"from the axis (constant z, one end at r = 0); got "
                    f"({z0:.6g}, {r0:.6g}) -> ({z1:.6g}, {r1:.6g}).")
            out.append((z0, max(abs(r0), abs(r1))))
        return sorted(out)

    def _signed_area(self):
        """Signed area of the endpoint polygon: > 0 counter-clockwise."""
        p = np.asarray(self._pts, dtype=float)
        if len(p) < 3:
            return 0.0
        z, r = p[:, 0], p[:, 1]
        return 0.5 * float(np.sum(z * np.roll(r, -1) - np.roll(z, -1) * r))

    # -- netgen.occ backend -------------------------------------------------

    def to_occ_face(self):
        """The profile as a netgen.occ face in the XY plane (x = z, y = r)."""
        from netgen.occ import (Segment, Wire, Face, Pnt, ArcOfCircle,
                                Ellipse, gp_Ax2d, gp_Pnt2d, gp_Dir2d, BezierCurve)

        if len(self._segs) < 3:
            raise ValueError("A profile needs at least 3 segments to bound a face.")
        edges = []
        for s in self._segs:
            p0 = self._pts[s['i0']]
            p1 = self._pts[s['i1']]
            if s['kind'] == 'arc':
                m = s['mid']
                edges.append(ArcOfCircle(Pnt(p0[0], p0[1], 0), Pnt(m[0], m[1], 0),
                                         Pnt(p1[0], p1[1], 0)))
            elif s['kind'] == 'ellipse':
                c, major, minor, xdir, t_lo, t_hi = self._ellipse_span(s, self._pts)
                ax = gp_Ax2d(gp_Pnt2d(c[0], c[1]), gp_Dir2d(xdir[0], xdir[1]))
                edges.append(Ellipse(ax, major, minor).Trim(t_lo, t_hi).Edge())
            elif s['kind'] == 'spline':
                poles = self._spline_poles(s, self._pts)
                if s['spline_kind'] == 'bezier':
                    edges.append(BezierCurve([Pnt(z, r, 0) for z, r in poles]))
                else:
                    for bez in self._bspline_to_bezier(poles, self._spline_degree(s, self._pts)):
                        edges.append(BezierCurve([Pnt(z, r, 0) for z, r in bez]))
            else:
                edges.append(Segment(Pnt(p0[0], p0[1], 0), Pnt(p1[0], p1[1], 0)))
        return Face(Wire(edges))

    def to_occ_pieces(self):
        """``[(material, face), ...]``: the profile face split by material region.

        With no regions this is the whole face as :attr:`default_material`.
        """
        face = self.to_occ_face()
        if not self._regions:
            return [(self.default_material, face)]
        # Profiles are traced clockwise; booleans on a clockwise face give pieces
        # whose shared edges do not glue, so orient it counter-clockwise first.
        if self._signed_area() < 0:
            face = face.Reversed()
        rest, pieces = face, []
        for reg in self._regions:
            rface = reg.to_occ_face()
            piece = rest * rface
            if not len(piece.faces):
                raise ValueError(
                    f"material region {reg.material!r} (z={reg.z}, r={reg.r}) does not "
                    f"overlap profile {self.name!r} (or is covered by an earlier region).")
            pieces.append((reg.material, piece))
            rest = rest - rface
        if len(rest.faces):
            pieces.append((self.default_material, rest))
        return pieces


def revolve(profile, port_prefix='port', wall_name='default', interface_name='interface'):
    """Sweep *profile* 360 degrees about the beam axis into a named solid.

    The beam axis is Z; the profile's z becomes Z and r the distance from it.
    Faces are named for the solver:

    - each ``'PMC'`` aperture becomes ``port1``, ``port2``, ... in increasing z;
    - a face shared by two material regions becomes *interface_name*;
    - every other face (the ``'PEC'`` wall) becomes *wall_name*.

    Solids carry their material name (``profile.default_material`` for the
    vacuum). Returns the netgen.occ shape.
    """
    from netgen.occ import Axis, X, Y, Glue

    corners = profile.stationary_corners()
    if corners:
        where = ', '.join('(%.6g, %.6g) mm' % (c[0] * 1e3, c[1] * 1e3) for c in corners[:3])
        raise ValueError(
            f"profile {profile.name!r} has {len(corners)} stationary corner(s), where "
            f"the spline tangent vanishes, near {where}. netgen cannot mesh the "
            "revolved surface there. This happens when a control polygon reverses on "
            "itself, e.g. a multicell B-spline built by repeating one cell's polygon; "
            "use kind='Bezier' (one curve per cell) instead.")
    apertures = profile.apertures()
    solids = []
    for material, face in profile.to_occ_pieces():
        # Revolve in the XY plane about X, then turn X onto Z.
        solid = face.Revolve(Axis((0, 0, 0), X), 360).Rotate(Axis((0, 0, 0), Y), -90)
        solid.mat(material)
        solids.append(solid)
    geo = solids[0] if len(solids) == 1 else Glue(solids)

    # OCC pads bounding boxes by its tolerance (~1e-7 m); allow for it.
    zs = [p[0] for p in profile.points]
    rs = [p[1] for p in profile.points]
    tol = 1e-6 * max(1.0, max(zs) - min(zs), max(rs))

    counts = {}
    for solid in getattr(geo, 'solids', [geo]):
        for f in solid.faces:
            counts[f] = counts.get(f, 0) + 1

    n_ports = 0
    for f in geo.faces:
        (x0, y0, z0), (x1, y1, z1) = f.bounding_box
        name = wall_name
        if counts.get(f, 1) > 1:
            name = interface_name
        elif z1 - z0 < tol:
            r_face = max(abs(x0), abs(x1), abs(y0), abs(y1))
            zc = 0.5 * (z0 + z1)
            for i, (z_ap, r_ap) in enumerate(apertures):
                if abs(zc - z_ap) < tol and abs(r_face - r_ap) < tol:
                    name = f'{port_prefix}{i + 1}'
                    n_ports += 1
                    break
        f.name = name
        if name.startswith(port_prefix):
            f.col = (1, 0, 0)
    if n_ports != len(apertures):
        warnings.warn(
            f"profile {profile.name!r}: found {n_ports} port face(s) for "
            f"{len(apertures)} aperture(s) after revolving; check the contour.",
            RuntimeWarning, stacklevel=2)
    return geo
