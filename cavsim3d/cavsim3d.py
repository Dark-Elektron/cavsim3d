"""Facade module re-exporting split components.

This module keeps the original single-file import path ``cavsim3d.cavsim3d`` but
the heavy implementation has been moved to submodules for clarity.
"""

from .config import settings

from .geometry import RWG, GMSHGeo, TESLA, STEPCavity
from .analytic import RWGAnalytic
from .models import Cavity, Concat

__all__ = [
    'settings',
    'RWG', 'GMSHGeo', 'TESLA', 'STEPCavity',
    'RWGAnalytic',
    'Cavity', 'Concat'
]


class STEPCavity:
    def __init__(self, filepath, unit='mm'):
        self.reader = STEPControl_Reader()
        self.status = self.reader.ReadFile(filepath)
        self.set_unit(unit)
        if self.status != IFSelect_RetDone:
            raise RuntimeError("STEP file could not be read.")

        # Transfer the roots to obtain a single compound shape
        self.reader.TransferRoots()
        self.shape = self.reader.OneShape()

        self.planes = []
        self.compound_geo = None
        self.mesh = None

    def add_planes(self, planes: list):
        for plane in planes:
            self.add_plane(plane)

    def set_unit(self, dim='mm'):
        if dim == 'mm':
            self.reader.SetSystemLengthUnit(1000) # units to mm
        elif dim == 'cm':
            self.reader.SetSystemLengthUnit(100)
        elif dim == 'm':
            self.reader.SetSystemLengthUnit(1)

    def add_plane(self, plane):
        self.planes.append(self.make_splitter_plane(*plane))

    def make_splitter_plane(self, pt1, pt2):
        x1, y1, z = pt1
        x2, y2, z = pt2
        W = BRepBuilderAPI_MakePolygon()
        W.Add(gp_Pnt(x1, y1, z))
        W.Add(gp_Pnt(x1, y2, z))
        W.Add(gp_Pnt(x2, y2, z))
        W.Add(gp_Pnt(x2, y1, z))
        W.Close()

        shape1_wire = W.Shape()
        shape1_face = BRepBuilderAPI_MakeFace(shape1_wire)

        return shape1_face

    def split(self):
        splitter = BOPAlgo_Splitter()
        splitter.SetNonDestructive(False)
        splitter.AddArgument(self.shape)  # object to cut
        if isinstance(self.planes, BRepBuilderAPI_MakeFace):
            splitter.AddTool(self.planes.Shape())  # tool means arguments are cut by this
        else:
            for tool in self.planes:
                splitter.AddTool(tool.Shape())  # tool means arguments are cut by this
        splitter.Perform()
        self.compound_geo = splitter.Shape()

        # save geo
        self.write_geo()

        # mesh compound geo
        self.build_mesh()

    def write_geo(self):
        # optional: write in AP203 (old) or AP214
        Interface_Static.SetCVal("write.step.schema", "AP214")

        writer = STEPControl_Writer()
        writer.Transfer(self.compound_geo, STEPControl_AsIs)
        writer.Write(fr"C:\Users\Soske\Documents\git_projects\cavsim2d\my_geometry.step")

    def build_mesh(self):
        geo = OCCGeometry(fr"C:\Users\Soske\Documents\git_projects\cavsim2d\my_geometry.step")
        self.geo = Glue([solid for solid in geo.solids])

        nsolids = len(self.geo.solids)
        for i, solid in enumerate(self.geo.solids):
            matname = f"subdomain_{i+1}"  # or any name you want
            self.geo.solids[i].mat(matname)
            self.geo.solids[i].faces.col = (i%nsolids, i%(nsolids-1), 1)
            # print(solid.bounding_box)
            self.geo.solids[i].faces.Max(Z).name = f'port{i+1}'
            if i == nsolids-1:
                self.geo.solids[i].faces.Min(Z).name = fr'port{i+2}'

        # Draw(geom)
        ngmesh = OCCGeometry(self.geo).GenerateMesh(maxh=0.05)

        # Convert to NGSolve mesh
        self.mesh = Mesh(ngmesh)
        print("materials", self.mesh.GetMaterials())
        print("boundaries", self.mesh.GetBoundaries())
        # Draw(ngmesh)

    def view(self, which='compound'):
        rnd = JupyterRenderer()
        if 'planes' in which:
            if isinstance(self.planes, BRepBuilderAPI_MakeFace):
                rnd.DisplayShape(self.planes.Shape(), render_edges=True)
            else:
                for plane in self.planes:
                    rnd.DisplayShape(plane.Shape(), render_edges=True)
        if 'model' in which:
            rnd.DisplayShape(self.shape, render_edges=True)

        if which == 'compound':
            rnd.DisplayShape(self.compound_geo, render_edges=True)

        rnd.Display()


class Cavity:
    def __init__(self, geo):
        self.fess = {}
        self.freqs = None
        self.geo = geo

        # define domains
        self.domain = [d for d in geo.mesh.GetMaterials() if 'subdomain' in d]
        self.ports = [p for p in geo.mesh.GetBoundaries() if 'port' in p]
        self.domain_port_map = self.assign_ports_to_domains(self.domain, self.ports)
        self.mesh = geo.mesh
        self.fesorder, self.bc = 3, 'default'

        # full order model dictionaries
        self.Ms, self.Ks, self.Bs, self.Ws, self.Zs = [dict() for _ in range(5)]

        # reduced order model dictionaries
        self.Wrs, self.Ards, self.QLinvs, self.Brds, self.Zrds = [dict() for _ in range(5)]

        self.description = {'K': 'Stiffness Matrix',
                            'M': 'Mass Matrix',
                            'Z': 'Z Parameters Dictionary',
                            'S': 'S Parameters Dictionary',}

    def analyse(self):
        for structure in self.geo:
            pass

    def port_excitations(self, space, mesh, order, bc):
        # eigenmode on faces marked as ports
        Einc, B, nports = {}, {}, 0

        for boundary_label in mesh.GetBoundaries():
            if 'port' in boundary_label:
                nports += 1
                fesport = space(mesh, order=order,
                                  dirichlet=bc,
                                  definedon=mesh.Boundaries(boundary_label))
                Einc[boundary_label], B[boundary_label] = self.port_eigenmodes(fesport, mesh, boundary_label)

        # Align all port modes to selected reference port. MODIFY LATER FOR INDEPENDENT PORT ORIENTATION
        ports = list(Einc.keys())
        nmodes = len(Einc[ports[0]])
        for mode_num in range(nmodes):
            ref_port = ports[0]
            ref_mode = Einc[ref_port][mode_num] # align all other port modes to modes at port 1
            vref = ref_mode.vec.FV().NumPy()
            for j, port in enumerate(ports):
                # Get vector as NumPy array
                vport = Einc[port][mode_num].vec.FV().NumPy()

                if port != ref_port:
                    if np.real(np.vdot(vref, vport)) < 0:
                        orient = -1
                        Einc[port][mode_num].vec.data = orient*Einc[port][mode_num].vec
                        B[port][mode_num] = orient*B[port][mode_num]

        return Einc, B, nports

    def port_eigenmodes(self, fesport, mesh, port, nmodes=1):
        u, v = fesport.TnT()
        a = BilinearForm(curl(u.Trace())*curl(v.Trace())*ds(port))
        m = BilinearForm(u.Trace()*v.Trace()*ds(port))
        apre = BilinearForm((curl(u).Trace()*curl(v).Trace() + u.Trace()*v.Trace())*ds(port))
        pre = Preconditioner(apre, type="direct", inverse="sparsecholesky")

        with TaskManager():
            a.Assemble()
            m.Assemble()
            apre.Assemble()

            G, fesh1 = fesport.CreateGradient()
            GT = G.CreateTranspose()
            math1 = GT @ m.mat @ G
            invh1 = math1.Inverse(inverse="sparsecholesky", freedofs=fesh1.FreeDofs())

            proj = IdentityMatrix(fesport.ndof) - G@invh1@GT@m.mat

            projpre = proj@pre
            evals, evecs = solvers.PINVIT(a.mat, m.mat, pre=projpre, num=nmodes+2, maxit=20,
                                          printrates=False)
        filt = np.array(evals) > 1
        evals = np.array(evals)[filt]
        evecs = np.array(evecs)[filt]
        freq_fes = []
        # evals[0] = 1  # <- replace nan with zero
        for i, lam in enumerate(evals):
            freq_fes.append(c0 * np.sqrt(np.abs(lam)) / (2 * np.pi) * 1e-6)
        # print(freq_fes)

        E, B = dict(), dict()
        for mode_num in range(nmodes):
            efield = GridFunction(fesport)
            efield.vec.data = evecs[mode_num]

            U = Integrate(InnerProduct(efield, efield), mesh, BND, definedon=mesh.Boundaries(port))
            efield.vec.data = efield.vec/np.sqrt(U)

            E[mode_num] = efield
            # Draw(efield, settings=settings)

            efield_mass_weighted = GridFunction(fesport)
            efield_mass_weighted.vec.data = sp.csr_matrix(m.mat.CSR())@evecs[mode_num]
            # efield_full.Set(efield, definedon=mesh.Boundaries(port))
            # efield_full.Set(-mu0*efield, definedon=mesh.Boundaries(port))

            B[mode_num] = efield_mass_weighted

        return E, B

    def assemble_matrices(self):
        # define fes on 3D domain
        for subdomain in self.mesh.GetMaterials():
            fes = HCurl(self.mesh, order=self.fesorder, dirichlet=self.bc, definedon=self.mesh.Materials(subdomain))
            self.fess[subdomain] = fes

            u, v = fes.TnT()
            r2 = BilinearForm(1/mu0*curl(u)*curl(v)*dx(subdomain))
            m2 = BilinearForm(eps0*u*v*dx(subdomain))
            with TaskManager():
                r2.Assemble()
                m2.Assemble()

            self.Ms[subdomain] = sp.csr_matrix(m2.mat.CSR()).copy()
            self.Ks[subdomain] = sp.csr_matrix(r2.mat.CSR()).copy()

        # get port excitations
        Lport, self.B, nport = self.port_excitations(HCurl, self.mesh, self.fesorder, self.bc)

        self._construct_b()

        # current matrices
        I, _, _ = self.excitation_matrix_from_ports()

        return self.Ms, self.Ks, self.Bs

    def assign_ports_to_domains(self, domains, ports):
        # REPLACE FUNCITON LATER. THIS IS ONLY VALID FOR 2-PORT DOMAINS
        if len(ports) != len(domains) + 1:
            raise ValueError("Number of ports must be number of domains + 1")

        mapping = {}
        for i, dom in enumerate(domains):
            mapping[dom] = [ports[i], ports[i+1]]

        return mapping

    def _construct_b(self):
        # if isinstance(self.B, dict):
        for subdomain, ports in self.domain_port_map.items():
            fes_subdomain = HCurl(self.mesh, order=self.fesorder, dirichlet=self.bc, definedon=subdomain)
            for port in ports:
                for mode in self.B[port].keys():
                    efield_subdomain = GridFunction(fes_subdomain)
                    efield_subdomain.Set(self.B[port][mode], definedon=self.mesh.Boundaries(port))
                    if subdomain not in self.Bs.keys():
                        self.Bs[subdomain] = []
                    self.Bs[subdomain].append(efield_subdomain.vec.FV().NumPy())

        keys = list(self.Bs.keys())
        for key in keys:
            self.Bs[key] = np.array(self.Bs[key]).T

    def excitation_matrix_from_ports(self):
        # Get ports in order of appearance
        self.ports = [b for b in self.mesh.GetBoundaries() if "port" in b]
        self.n_ports = len(self.ports)

        # Identity matrix for excitation
        self.I = np.eye(self.n_ports, dtype=int)

        # Map port names to row indices
        self.port_indices = {port: i for i, port in enumerate(self.ports)}

        return self.I, self.ports, self.port_indices

    def solve_FD(self, fmin, fmax, nsamples=1000):
        # check that fesorder and boundary conditions are set
        freqs = np.linspace(fmin, fmax, nsamples)*1e9
        self.Zs['freqs'] = freqs

        # get port excitations
        Lport, self.B, nport = self.port_excitations(HCurl, self.mesh, self.fesorder, self.bc)
        ports = list(Lport.keys())
        pairs = [[ports[i], ports[i+1]] for i in range(len(ports)-1)]

        for nn, (subdomain, ports) in enumerate(zip(self.mesh.GetMaterials(), pairs)):
            W = []
            for kk, freq in enumerate(freqs):

                self.Zs[subdomain] = {}
                w = 2*pi*freq

                # define fes on 3d domain
                fes = HCurl(self.mesh, order=self.fesorder, dirichlet=self.bc, definedon=self.mesh.Materials(subdomain))
                u, v = fes.TnT()

                # build forms
                m = BilinearForm(1/mu0*curl(u)*curl(v)*dx(subdomain) - w**2*eps0*u*v*dx(subdomain))
                # matrices for ssc
                with TaskManager():
                    m.Assemble()

                # loop for rhs, lhs remains the same
                # calculate impedance for all ports, all modes
                for pm, port_m in enumerate(ports):
                    L_m = Lport[port_m]
                    for mode_m, l_m in L_m.items():
                        # modal forcing: i*k0*Z0 ∫ E_inc · v ds(port)
                        f = LinearForm(fes)
                        f += w * InnerProduct(Lport[port_m][mode_m], v.Trace()) * ds(port_m)

                        with TaskManager():
                            f.Assemble()
                            E = GridFunction(fes)
                            E.vec.data = m.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec

                            W.append(E.vec.FV().NumPy()) # REMEMBER LATER, NO FACTOR OF 1j INCLUDED HERE SO X=E/1j
                            # Xfes = GridFunction(fes)
                            # Xfes.vec.data = BaseVector(E.vec.FV().NumPy())
                            # X = BoundaryFromVolumeCF(Xfes*1j)
                            # Draw(Norm(X), self.mesh, settings=settings)

                            E = BoundaryFromVolumeCF(1j*E)

                            # print(subdomain, pm, ports)
                            # Draw(Norm(E), self.mesh, settings=settings)

                        for pn, port_n in enumerate(ports):
                            L_n = Lport[port_n]
                            for mode_n, l_n in L_n.items():
                                self.Zs[subdomain][f'{pn+1}({mode_n+1}){pm+1}({mode_m+1})'] = Integrate(InnerProduct(E, l_n),
                                                                                       self.mesh,
                                                                                       BND, definedon=self.mesh.Boundaries(port_n))
            W = np.array(W).T.copy()
            self.Ws[subdomain] = W

    def rom(self, tol=1e-1):
        for subdomain, W in self.Ws.items():
            print('Subdomain', subdomain, W.shape)
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            # fig,ax = plt.subplot_mosaic([[1]], layout='constrained', figsize=(8,4))
            # ax[1].plot(S)
            # ax[1].set_yscale('log')
            # plt.grid()
            # plt.show()

            print('\tDecomp shapes', U.shape, S.shape, Vt.shape)
            r = len(S)
            r = 20
            Sr = S[:r]
            Vtr = Vt[:r, :]
            Wr = U[:, :r]
            print('\tTrunc shapes', Wr.shape, Sr.shape, Vtr.shape, r)

            # self._test_reconstruction(W, Wr, Sr, Vtr)
            Mr, Rr = Wr.T@self.Ms[subdomain]@Wr, Wr.T@self.Ks[subdomain]@Wr

            Mr = (Mr + Mr.T)/2
            Rr = (Rr + Rr.T)/2
            # self._check_reduced_model(Wr, Rr, Mr)

            lam, Q = sl.eigh(Mr)

            inv_sqrt_lam = 1/np.sqrt(lam)
            QLinv = Q@np.diag(inv_sqrt_lam)

            Ard = QLinv.T@Rr@QLinv
            Ard = (Ard + Ard.T)/2

            Brd = QLinv.T @ Wr.T @ self.Bs[subdomain]

            self.Wrs[subdomain] = Wr
            self.Ards[subdomain] = Ard
            self.QLinvs[subdomain] = QLinv
            self.Brds[subdomain] = Brd
        return self.Wrs, self.Ards, self.QLinvs, self.Brds

    def check_eigs_full_system(self):
        # for testing only for small sparse matrices
        for subdomain, M in self.Ms.items():
            K = self.Ks[subdomain]
            fes = self.fess[subdomain]
            freedofs = fes.FreeDofs()

            evals, evecs = sp.linalg.eigs(K[freedofs, :][:, freedofs], k=20, M=M[freedofs, :][:, freedofs], sigma=1e19)
            sort = np.argsort(evals)
            evals_free = evals[sort]
            evecs_free = evecs[:, sort]
            mask = evals_free > 1e18
            evals_free = evals_free[mask]
            evecs_free = evecs_free[:, mask]
            print(f"Full system eigenvalues:: {subdomain}\n\t", evals_free)

    def check_eigs_reduced_system(self, mode=None):
        for subdomain, Ard in self.Ards.items():
            Wr = self.Wrs[subdomain]
            QLinv = self.QLinvs[subdomain]

            lrp, xrp = sl.eigh(Ard)
            # sort eigenmode and eigenvalues
            idx_sort = np.argsort(np.abs(lrp))
            lrp = lrp[idx_sort]
            xrp = xrp[:, idx_sort]
            print(f"Reduced system eigenvalues:: {subdomain}\n\t", lrp)

            if mode is not None:
                fes = HCurl(self.mesh, order=self.fesorder, dirichlet=self.bc)
                Erd = GridFunction(fes)
                x = Wr @ QLinv @ xrp[:, 0]
                Erd.vec.data = x
                Erd = BoundaryFromVolumeCF(Erd)
                Draw(Norm(Erd), self.mesh, settings=settings)

    def prepare(self):
        self.assemble_matrices() # assemble system matrices
        self.solve_FD(1, 3, 10) # to get snapshots
        self.rom() # to get reduced system matrices
        print()
        self.check_eigs_full_system()
        print()
        self.check_eigs_reduced_system()
        # self.zreduced(1, 3, 100)


# class Structure:
#     def __init__(self, geo):
#         self.freqs = None
#         self.geo = geo
#         self.mesh = geo.mesh
#         self.fesorder, self.bc = 3, geo.bc
#         self.M, self.K, self.B = None, None, None
#         self.Z, self.S, self.Wfull = {}, {}, []
#         self.Zrd = []
#         self.global_offset = 0
#         self.description = {'K': 'Stiffness Matrix',
#                             'M': 'Mass Matrix',
#                             'Z': 'Z Parameters Dictionary',
#                             'S': 'S Parameters Dictionary',}
#
#     def excitation_matrix_from_ports(self):
#         # Get ports in order of appearance
#         self.ports = [b for b in self.mesh.GetBoundaries() if "port" in b]
#         self.n_ports = len(self.ports)
#
#         # Identity matrix for excitation
#         self.I = np.eye(self.n_ports, dtype=int)
#
#         # Map port names to row indices
#         self.port_indices = {port: i for i, port in enumerate(self.ports)}
#
#         return self.I, self.ports, self.port_indices
#
#     def _check_reduced_model(self, Wr, Rr, Mr, draw=False):
#         print('============Checking reduced system eigenvalue============')
#         # check eigenvalue sof reduced system
#         evals_r, evecs_r = sl.eigh(Rr, Mr)
#         # evals_r, evecs_r = sp.linalg.eigs(Rr, k=10, M=Mr)
#
#         sort = np.argsort(evals_r)
#         evals_r = evals_r[sort]
#         evecs_r = evecs_r[sort]
#         print('evals_rd', evals_r)
#
#         mode = 0
#         xr = Wr@evecs_r
#         xrm = xr[:, mode]
#         if draw:
#             fes = HCurl(self.mesh, order=self.fesorder, dirichlet=self.bc)
#             Er = GridFunction(fes)
#             Er.vec.data = BaseVector(xrm)
#             Er = BoundaryFromVolumeCF(Er)
#             Draw(Norm(Er), self.mesh, settings=settings)
#         print('============Done checking reduced system eigenvalue============\n')
#         return xr
#
#     def _test_reconstruction(self, W, Wr, Sr, Vtr):
#         # test reconstruction
#         print('============Testing snap reconstruciton============')
#         Esnap = W
#         Erecon = Wr@np.diag(Sr)@Vtr
#         err = np.linalg.norm(Erecon - Esnap) / np.linalg.norm(Esnap)
#         print("Snap reconstruction error", err)
#         print('============Done testing snap reconstruciton============\n')
#
#     def port_excitations(self, space, mesh, order, bc):
#         # eigenmode on faces marked as ports
#         fesfull = space(mesh, order=order, dirichlet=bc)
#         Einc, B, nports = {}, {}, 0
#
#         for boundary_label in mesh.GetBoundaries():
#             if 'port' in boundary_label:
#                 nports += 1
#                 fesport = space(mesh, order=order,
#                                   dirichlet=bc,
#                                   definedon=mesh.Boundaries(boundary_label))
#                 Einc[boundary_label], B[boundary_label] = self.port_eigenmodes(fesport, fesfull, mesh, boundary_label)
#
#         # Align all port modes to selected reference port. MODIFY LATER FOR INDEPENDENT PORT ORIENTATION
#         ports = list(Einc.keys())
#         nmodes = len(Einc[ports[0]])
#         for mode_num in range(nmodes):
#             ref_port = ports[0]
#             ref_mode = Einc[ref_port][mode_num] # align all other port modes to modes at port 1
#             vref = ref_mode.vec.FV().NumPy()
#             for j, port in enumerate(ports):
#                 # Get vector as NumPy array
#                 vport = Einc[port][mode_num].vec.FV().NumPy()
#
#                 if port != ref_port:
#                     if np.real(np.vdot(vref, vport)) < 0:
#                         orient = -1
#                         Einc[port][mode_num].vec.data = orient*Einc[port][mode_num].vec
#                         B[port][mode_num] = orient*B[port][mode_num]
#
#         return Einc, B, nports
#
#     def port_eigenmodes(self, fesport, fesfull, mesh, port, nmodes=1):
#         u, v = fesport.TnT()
#         a = BilinearForm(curl(u.Trace())*curl(v.Trace())*ds(port))
#         m = BilinearForm(u.Trace()*v.Trace()*ds(port))
#         apre = BilinearForm((curl(u).Trace()*curl(v).Trace() + u.Trace()*v.Trace())*ds(port))
#         pre = Preconditioner(apre, type="direct", inverse="sparsecholesky")
#
#         with TaskManager():
#             a.Assemble()
#             m.Assemble()
#             apre.Assemble()
#
#             G, fesh1 = fesport.CreateGradient()
#             GT = G.CreateTranspose()
#             math1 = GT @ m.mat @ G
#             invh1 = math1.Inverse(inverse="sparsecholesky", freedofs=fesh1.FreeDofs())
#
#             proj = IdentityMatrix(fesport.ndof) - G@invh1@GT@m.mat
#
#             projpre = proj@pre
#             evals, evecs = solvers.PINVIT(a.mat, m.mat, pre=projpre, num=nmodes+2, maxit=20,
#                                           printrates=False)
#         filt = np.array(evals) > 1
#         evals = np.array(evals)[filt]
#         evecs = np.array(evecs)[filt]
#         freq_fes = []
#         # evals[0] = 1  # <- replace nan with zero
#         for i, lam in enumerate(evals):
#             freq_fes.append(c0 * np.sqrt(np.abs(lam)) / (2 * np.pi) * 1e-6)
#         # print(freq_fes)
#
#         E, B = dict(), dict()
#         for mode_num in range(nmodes):
#             efield = GridFunction(fesport)
#             efield.vec.data = evecs[mode_num]
#
#             U = Integrate(InnerProduct(efield, efield), mesh, BND, definedon=mesh.Boundaries(port))
#             efield.vec.data = efield.vec/np.sqrt(U)
#
#             E[mode_num] = efield
#             # Draw(efield, settings=settings)
#
#             efield_mass_weighted = GridFunction(fesport)
#             efield_mass_weighted.vec.data = sp.csr_matrix(m.mat.CSR())@evecs[mode_num]
#             efield_full = GridFunction(fesfull)
#             efield_full.Set(efield_mass_weighted, definedon=mesh.Boundaries(port))
#             # efield_full.Set(efield, definedon=mesh.Boundaries(port))
#             # efield_full.Set(-mu0*efield, definedon=mesh.Boundaries(port))
#
#             B[mode_num] = efield_full.vec.FV().NumPy()
#
#         return E, B
#
#     def check_port_normalisation(self, Einc):
#         # confirm normalisation
#         for port, efields in Einc.items():
#             for mode, efield in efields.items():
#                 int_LmLm = Integrate(InnerProduct(efield, efield), self.mesh, BND, definedon=self.mesh.Boundaries(port))
#                 print('Confirm normalisation: ', int_LmLm)
#
#     def set_fes_parameters(self, fesorder, bc='default'):
#         self.fesorder = fesorder
#         self.bc = bc
#
#     def assemble_matrices(self):
#         # define fes on 3D domain
#         fes = HCurl(self.geo.mesh, order=self.fesorder, dirichlet=self.bc)
#         u, v = fes.TnT()
#         r2 = BilinearForm(1/mu0*curl(u)*curl(v)*dx)
#         m2 = BilinearForm(eps0*u*v*dx)
#         with TaskManager():
#             r2.Assemble()
#             m2.Assemble()
#
#         self.M = sp.csr_matrix(m2.mat.CSR()).copy()
#         self.K = sp.csr_matrix(r2.mat.CSR()).copy()
#
#         # get port excitations
#         Lport, self.B, nport = self.port_excitations(HCurl, self.mesh, self.fesorder, self.bc)
#         self._construct_b()
#
#         # current matrices
#         I, _, _ = self.excitation_matrix_from_ports()
#
#         return self.M, self.K, self.B
#
#     def _construct_b(self):
#         if isinstance(self.B, dict):
#             Bx = []
#             for port, modes in self.B.items():
#                 for mode in modes:
#                     Bx.append(self.B[port][mode])
#
#             self.B = np.array(Bx).T
#
#     def get_snapshots(self, fmin, fmax, nsamples=1000):
#         self.solve_FD(fmin, fmax, nsamples)
#         return self.Wfull
#
#     def solve_FD(self, fmin, fmax, nsamples=1000):
#         # check that fesorder and boundary conditions are set
#         freqs = np.linspace(fmin, fmax, nsamples)*1e9
#         self.Z['freqs'] = freqs
#
#         # get port excitations
#         Lport, self.B, nport = self.port_excitations(HCurl, self.mesh, self.fesorder, self.bc)
#
#         for kk, freq in enumerate(freqs):
#             w = 2*pi*freq
#             k = w/c0
#
#             # define fes on 3d domain
#             fes = HCurl(self.mesh, order=self.fesorder, dirichlet=self.bc)
#             u, v = fes.TnT()
#
#             # build forms
#             m = BilinearForm(1/mu0*curl(u)*curl(v)*dx - w**2*eps0*u*v*dx)
#             # matrices for ssc
#             with TaskManager():
#                 m.Assemble()
#
#             # loop for rhs, lhs remains the same
#             # calculate impedance for all ports, all modes
#             for pm, (port_m, L_m) in enumerate(Lport.items()):
#                 for mode_m, l_m in L_m.items():
#                     # modal forcing: i*k0*Z0 ∫ E_inc · v ds(port)
#                     f = LinearForm(fes)
#                     f += w * InnerProduct(Lport[port_m][mode_m], v.Trace()) * ds(port_m)
#
#                     with TaskManager():
#                         f.Assemble()
#                         E = GridFunction(fes)
#                         E.vec.data = m.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec
#
#                         self.Wfull.append(E.vec.FV().NumPy()) # REMEMBER LATER, NO FACTOR OF 1j INCLUDED HERE SO X=E/1j
#                         # Xfes = GridFunction(fes)
#                         # Xfes.vec.data = BaseVector(E.vec.FV().NumPy())
#                         # X = BoundaryFromVolumeCF(Xfes*1j)
#                         # Draw(Norm(X), self.mesh, settings=settings)
#
#                         E = BoundaryFromVolumeCF(1j*E)
#
#                     for pn, (port_n, L_n) in enumerate(Lport.items()):
#                         for mode_n, l_n in L_n.items():
#                             if kk == 0:
#                                 self.Z[f'{pn+1}({mode_n+1}){pm+1}({mode_m+1})'] = []
#
#                             self.Z[f'{pn+1}({mode_n+1}){pm+1}({mode_m+1})'].append(Integrate(InnerProduct(E, l_n),
#                                                                                    self.mesh,
#                                                                                    BND, definedon=self.mesh.Boundaries(port_n)))
#         # post
#         self._construct_b()
#         self.Wfull = np.array(self.Wfull).T.copy()
#
#     def check_eigs_reduced_system(self, mode=None):
#         lrp, xrp = sl.eigh(self.Ard)
#         # sort eigenmode and eigenvalues
#         idx_sort = np.argsort(np.abs(lrp))
#         lrp = lrp[idx_sort]
#         xrp = xrp[:, idx_sort]
#         print("Reduced system eigenvalues", lrp)
#
#         if mode is not None:
#             fes = HCurl(self.mesh, order=self.fesorder, dirichlet=self.bc)
#             Erd = GridFunction(fes)
#             x = self.Wr @ self.QLinv @ xrp[:, 0]
#             Erd.vec.data = x
#             Erd = BoundaryFromVolumeCF(Erd)
#             Draw(Norm(Erd), self.mesh, settings=settings)
#
#     def zreduced(self, fmin, fmax, nsamples=1000):
#         freqs = np.linspace(fmin, fmax, nsamples)*1e9
#         self.freqZrd = {'freqs': freqs}
#
#         for kk, freq in enumerate(freqs):
#             w = 2 * pi * freq
#             lhs = self.Ard - w**2 * np.eye(self.Ard.shape[0])
#             # get excitation matrices
#             I, _, _ = self.excitation_matrix_from_ports()
#             rhs = w * self.Brd @ I # pay attention to the 1j removed from here always, fes is a real space
#
#             x_ = np.linalg.solve(lhs, rhs)
#             self.x_ = np.linalg.solve(lhs, rhs) # TEMPORARY FOR TESTING
#             xfull = self.Wr @ self.QLinv @ x_
#             if kk == -1:
#                 fes = HCurl(self.mesh, order=self.fesorder, dirichlet=self.bc)
#                 Ef = GridFunction(fes)
#                 Ef.vec.data = xfull[:, 0].ravel() # GridFunction vec.data takes a 1d array, must ravel
#                 Ef = BoundaryFromVolumeCF(Ef*1j)
#                 Draw(Norm(Ef), self.mesh, settings=settings)
#
#             z = 1j * self.Brd.T @ x_ # pay attention to the 1j always
#             self.Zrd.append(z)
#
#         self.Zrd = np.array(self.Zrd)
#         self.freqZrd['Z'] = self.Zrd
#         return self.Zrd
#
#     def z2s(self, Z, Z0):
#         I = np.eye(Z.shape[0])
#         D = np.sqrt(Z0) * I
#         Dinv = 1/np.sqrt(Z0) * I
#         A = Dinv@Z@Dinv
#         S = np.linalg.inv(A + I)@(A - I)
#         return S
#
#     def sparameters(self):
#         # ind = 900
#         # Zmat = np.array([[Z11_a[ind], Z21_a[ind]], [Z21_a[ind], Z11_a[ind]]])
#         # S = z2s(Zmat, ZTEs[ind])
#         # print(S)
#         freqs = Z['freqs']
#         S11 = []
#         for ind in range(0, len(freqs)):
#             Zmat = np.array([[Z11[ind], Z21[ind]], [Z21[ind], Z11[ind]]])
#             # print(Zmat)
#             S = self.z2s(Zmat, self.ZTEs[ind])
#             S11.append(S[0, 1])
#
#         fig, ax = plt.subplot_mosaic([[1]], layout='constrained', figsize=(8,4))
#         ax[1].plot(freqs, 20*np.log10(np.abs(S11)), marker='o', label='|S21| [dB] numerical')
#         plt.legend()
#
#         fig, ax = plt.subplot_mosaic([[1]], layout='constrained', figsize=(8,4))
#         ax[1].plot(freqs, np.angle(S11), marker='o', label='|S21| [dB] numerical')
#         plt.legend()
#
#     def rom(self, tol=1e-1):
#         U, S, Vt = np.linalg.svd(self.Wfull, full_matrices=False)
#         # fig,ax = plt.subplot_mosaic([[1]], layout='constrained', figsize=(8,4))
#         # ax[1].plot(S)
#         # ax[1].set_yscale('log')
#         # plt.grid()
#         # plt.show()
#
#         print('Decomp shapes', U.shape, S.shape, Vt.shape)
#         r = len(S[S>tol])
#         Sr = S[:r]
#         Vtr = Vt[:r, :]
#         Wr = U[:, :r]
#         print('Trunc shapes', Wr.shape, Sr.shape, Vtr.shape, r)
#
#         # self._test_reconstruction(W, Wr, Sr, Vtr)
#         Mr, Rr = Wr.T@self.M@Wr, Wr.T@self.K@Wr
#
#         Mr = (Mr + Mr.T)/2
#         Rr = (Rr + Rr.T)/2
#         self._check_reduced_model(Wr, Rr, Mr)
#
#         lam, Q = sl.eigh(Mr)
#
#         inv_sqrt_lam = 1/np.sqrt(lam)
#         QLinv = Q@np.diag(inv_sqrt_lam)
#
#         Ard = QLinv.T@Rr@QLinv
#         Ard = (Ard + Ard.T)/2
#         Brd = QLinv.T @ Wr.T @ self.B
#
#         self.Wr, self.Ard, self.QLinv, self.Brd = Wr, Ard, QLinv, Brd
#         return Wr, Ard, QLinv, Brd
#
#     def plotZ(self, which, ax=None):
#         from scipy.signal import find_peaks
#         # Z parameters
#         if which == 'full':
#             freqs = self.Z['freqs']
#             Z11 = self.Z['1(1)1(1)']
#             Z21 = self.Z['1(1)2(1)']
#         elif which == 'reduced':
#             freqs = self.freqZrd['freqs']
#             Z11 = self.freqZrd['Z'][:, 0, 0]
#             Z21 = self.freqZrd['Z'][:, 1, 0]
#         else:
#             freqs = self.Z['freqs']
#             Z11 = self.Z['1(1)1(1)']
#             Z21 = self.Z['1(1)2(1)']
#
#         # plot z parameters
#         if ax is None:
#             fig, ax = plt.subplot_mosaic([[1, 2], [3, 4]], layout='constrained', figsize=(10,8))
#         ax[1].plot(freqs, 20*np.log10(np.abs(Z11)), marker='o', label=f'|z11| [dB] {which}', mfc='none', lw=0)
#         ax[2].plot(freqs, 20*np.log10(np.abs(Z21)), marker='o', label=f'|z12| [dB] {which}', mfc='none', lw=0)
#         ax[1].set_ylabel('|z11| [dB]')
#         ax[1].set_xlabel('freq [GHz]')
#         ax[2].set_xlabel('freq [GHz]')
#         ax[2].set_ylabel('|z12| [dB]')
#         ax[1].legend()
#         ax[2].legend()
#
#         # plot phase
#         ax[3].plot(freqs, np.angle(Z11), marker='o', label=fr'$\angle$ z11 {which}', mfc='none', lw=0)
#         ax[4].plot(freqs, np.angle(Z21), marker='o', label=fr'$\angle$ z12 {which}', mfc='none', lw=0)
#         ax[3].set_xlabel('freq [GHz]')
#         ax[4].set_ylabel(r'$\angle$ z11 [deg]')
#         ax[4].set_xlabel('freq [GHz]')
#         ax[4].set_ylabel(r'$\angle$ z11 [deg]')
#         ax[3].legend()
#         ax[4].legend()
#
#         peaks, _ = find_peaks(20*np.log10(np.abs(Z11)))
#         print(freqs[peaks])
#         # save Z parameters
#         # import pickle
#         # with open('Zparametes_coarse_mesh.pickle', 'wb') as handle:
#         #     pickle.dump(Z, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         return ax
#
#     def prepare(self):
#         self.assemble_matrices() # assemble system matrices
#         self.solve_FD(1, 3, 10) # to get snapshots
#         self.rom() # to get reduced system matrices
#         self.check_eigs_reduced_system()
#         self.zreduced(1, 3, 100)
#

class Concat:
    def __init__(self, structs):
        if isinstance(structs, Structure):
            self.structs = structs
            self.assign_global_port()
        else:
            self.structs = structs

        self.geo = None
        self.mesh = None
        self.Wfull = None
        self.Zc, self.Zrd = [], []

    def glue_geo(self):
        offset = 0
        geo_list = []
        for struct in self.structs:
            if offset == 0:
                geo_list.append(struct.geo.geo)
            else:
                geo = struct.geo.geo.Move((0, 0, offset))
                geo_list.append(geo)
            offset += struct.geo.L

        self.geo = Glue(geo_list)
        self.mesh = Mesh(OCCGeometry(self.geo).GenerateMesh(maxh=0.05))
        Draw(self.mesh)

    def concat(self, connections):
        """
        Build permutation matrix that moves internal ports first.
        connections = [ ((sA, "portX"), (sB, "portY")), ... ]
        """
        # Total number of global ports
        self.nconnections = len(connections)
        total_ports = sum(len(s.ports) for s in self.structs)

        # Collect internal global indices
        internal = set()

        for (sA, pA), (sB, pB) in connections:
            gA = sA.global_offset + sA.port_indices[pA]
            gB = sB.global_offset + sB.port_indices[pB]
            internal.add(gA)
            internal.add(gB)

        internal = sorted(internal)

        # External ports = rest
        all_ports = set(range(total_ports))
        external = sorted(all_ports - set(internal))

        #  Final permutation list
        # Internal ports first, then external
        self.nint, self.next = len(internal), len(external)
        perm = internal + external
        self.permutation = perm

        # Create full permutation matrix
        PT = np.zeros((total_ports, total_ports), dtype=int)
        for new_pos, old_pos in enumerate(perm):
            PT[new_pos, old_pos] = 1

        self.PT = PT
        self.P = PT.T
        return self.P

    def assign_global_port(self):
        global_ports = []
        offset = 0
        for struct in self.structs:
            struct.global_offset = offset
            offset += len(struct.ports)

    def build_global_excitation(self):
        #  collect local blocks as 2D numpy arrays
        blocks = []
        for struct in self.structs:
            I_local = np.asarray(struct.I)

            # treat 1D as column vector
            if I_local.ndim == 1:
                I_local = I_local.reshape((-1, 1))

            # ensure correct number of rows
            n_ports = len(struct.ports)
            if I_local.shape[0] != n_ports:
                raise ValueError(
                    f"struct.I for {struct} has shape {I_local.shape}, "
                    f"expected ({n_ports}, _)"
                )

            blocks.append(I_local)

        # --- pad each block to the same number of columns ---
        max_cols = max(block.shape[1] for block in blocks)
        blocks_padded = []
        for B in blocks:
            if B.shape[1] < max_cols:
                pad = np.zeros((B.shape[0], max_cols - B.shape[1]), dtype=B.dtype)
                B = np.hstack([B, pad])
            blocks_padded.append(B)

        # --- vertical stack, NO permutation ---
        I_global = np.vstack(blocks_padded)

        self.I_global = I_global
        return I_global

    # def build_F(self):
    #     diag = []
    #     for _ in range(self.nconnections):
    #         diag.extend([1, -1])
    #     return np.diag(diag)

    def build_F(self, dtype=float):
        F = np.zeros((2 * self.nconnections, self.nconnections), dtype=dtype)
        for j in range(self.nconnections):
            F[2*j : 2*j+2, j] = [1, -1]
        return F

    def build_matrices(self):
        # concatenation
        sArds, sBrds, sxtest = [], [], []
        # for s in self.structs:
        #     sArds.append(s.Ard)
        #     sBrds.append(s.Brd)
        #     # print('sBrd',s.Brd.shape, s.Brd)
        #     sxtest.append(s.x_)

        Ab = sl.block_diag(self.structs.Ards['subdomain_1'], self.structs.Ards['subdomain_2'])
        Bb = sl.block_diag(self.structs.Brds['subdomain_1'], self.structs.Brds['subdomain_2'])
        print('Ab, Bb', Ab.shape, Bb.shape)

        # P = self.concat(connections)
        self.P = np.array([[0, 0, 1, 0],
                     [1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]])  # manually build permutation matrix

        # Ib = self.build_global_excitation()
        self.Ib = np.array([[1, 0],
                       [0, 1],
                       [1, 0],
                       [0, 1]])  # manually build current matrix
        Isrt = self.P.T@self.Ib
        # printf('\nIsrt\n', Isrt)

        self.nint, self.next, self.nconnections = 2, 2, 1  # manually enter number of internal and external nodes and number of connections
        BbP = Bb@self.P
        Bb1 = BbP[:, 0:self.nint]
        # print('\nBb1\n', Bb1)
        Bb2 = BbP[:, self.nint:]
        self.Iint = Isrt[0:self.nint, :]
        self.Iext = Isrt[self.nint:, :]
        self.F = self.build_F()
        F = self.build_F()

        # # CHECK KIRCHOFF, should be zero
        # print('as', F.T.shape, Bb1.shape, xtest.shape)
        # zero = F.T@Bb1.T@xtest
        # self.Bb1 = Bb1
        # self.xtest = xtest
        # print('check F.T@Bb1.T@xb: ', zero)

        Bb1F = Bb1@F
        # print('\nBb1F\n', Bb1F)
        K = np.eye(Bb1F.shape[0]) - Bb1F@np.linalg.inv(Bb1F.T@Bb1F)@Bb1F.T
        print('check K idempotency: ', np.max(K - K.T@K))

        M = sl.null_space(Bb1F.T)
        KM = K@M
        print("KM.T, Ab, KM", KM.T.shape, Ab.shape, KM.shape)
        self.Ac = KM.T@Ab@KM
        self.Bc = KM.T@Bb2

        # check eigenvalues
        # self.check_eigs_reduced_system()

    # check eigenvalues
    def check_eigs_system(self, A, mode=None):
        lrp, xrp = sl.eigh(A)
        # sort eigenmode and eigenvalues
        idx_sort = np.argsort(np.abs(lrp))
        lrp = lrp[idx_sort]
        xrp = xrp[:, idx_sort]
        print('Matrix size ', A.shape)
        print("System eigenvalues", lrp)

    def solve_FD(self, fmin, fmax, nsamples=1000):
        freqs = np.linspace(fmin, fmax, nsamples)*1e9
        self.freqZrd = {'freqs': freqs}

        for kk, freq in enumerate(freqs):
            w = 2 * pi * freq
            lhs = -self.Ac - (w / c0) ** 2 * np.eye(self.Ac.shape[0])
            # get excitation matrices
            rhs = w / c0 * Z0 * self.Bc @ self.Iext # pay attention to the 1j removed from here always, fes is a real space

            x_ = np.linalg.solve(lhs, rhs)
            # self.x_ = np.linalg.solve(lhs, rhs) # TEMPORARY FOR TESTING

            if self.Wfull is None:
                self.Wfull = x_
            else:
                self.Wfull = np.hstack([self.Wfull, x_])

        return self.Wfull

    def zreduced(self, fmin, fmax, nsamples=1000):
        freqs = np.linspace(fmin, fmax, nsamples)*1e9
        self.freqZrd = {'freqs': freqs}
        for kk, freq in enumerate(freqs):
            w = 2 * pi * freq
            lhs = self.Ard - w**2 * np.eye(self.Ard.shape[0])
            # get excitation matrices
            rhs = w * self.Brd @ self.Iext # pay attention to the 1j removed from here always, fes is a real space

            x_ = np.linalg.solve(lhs, rhs)
            self.x_ = np.linalg.solve(lhs, rhs) # TEMPORARY FOR TESTING
            # xfull = self.Wr @ x_
            # if kk == -1:
            #     fes = HCurl(self.mesh, order=self.fesorder, dirichlet=self.bc)
            #     Ef = GridFunction(fes)
            #     Ef.vec.data = xfull[:, 0].ravel() # GridFunction vec.data takes a 1d array, must ravel
            #     Ef = BoundaryFromVolumeCF(Ef*1j)
            #     Draw(Norm(Ef), rwg.mesh, settings=settings)

            z = 1j*self.Brd.T @ x_ # pay attention to the 1j always
            self.Zrd.append(z)

        self.Zrd = np.array(self.Zrd)
        self.freqZrd['Z'] = self.Zrd
        return self.Zrd

    def zfull(self, fmin, fmax, nsamples=1000):
        freqs = np.linspace(fmin, fmax, nsamples)*1e9
        self.freqZc = {'freqs': freqs}

        for kk, freq in enumerate(freqs):
            w = 2 * pi * freq
            lhs = self.Ac - w** 2 * np.eye(self.Ac.shape[0])
            # get excitation matrices
            rhs = w * self.Bc @ self.Iext # pay attention to the 1j removed from here always, fes is a real space

            x_ = np.linalg.solve(lhs, rhs)

            # xfull = self.Wr @ x_
            # if kk == -1:
            #     fes = HCurl(self.mesh, order=self.fesorder, dirichlet=self.bc)
            #     Ef = GridFunction(fes)
            #     Ef.vec.data = xfull[:, 0].ravel() # GridFunction vec.data takes a 1d array, must ravel
            #     Ef = BoundaryFromVolumeCF(Ef*1j)
            #     Draw(Norm(Ef), rwg.mesh, settings=settings)

            z = 1j*self.Bc.T @ x_ # pay attention to the 1j always
            self.Zc.append(z)

        self.Zc = np.array(self.Zc)
        self.freqZc['Z'] = self.Zc
        return self.Zc

    def plotZ(self, which, ax=None):
        from scipy.signal import find_peaks
        # Z parameters
        if which == 'full':
            freqs = self.freqZc['freqs']
            Z11 = self.freqZc['Z'][:, 0, 0]
            Z21 = self.freqZc['Z'][:, 1, 0]
        elif which == 'reduced':
            freqs = self.freqZrd['freqs']
            Z11 = self.freqZrd['Z'][:, 0, 0]
            Z21 = self.freqZrd['Z'][:, 1, 0]
        else:
            freqs = self.freqZc['freqs']
            Z11 = self.freqZc['Z'][:, 0, 0]
            Z21 = self.freqZc['Z'][:, 1, 0]

        # plot z parameters
        if ax is None:
            fig, ax = plt.subplot_mosaic([[1, 2], [3, 4]], layout='constrained', figsize=(10,8))
        ax[1].plot(freqs, 20*np.log10(np.abs(Z11)), marker='o', label=f'|z11| [dB] {which}', mfc='none', lw=0)
        ax[2].plot(freqs, 20*np.log10(np.abs(Z21)), marker='o', label=f'|z12| [dB] {which}', mfc='none', lw=0)
        ax[1].set_ylabel('|z11| [dB]')
        ax[1].set_xlabel('freq [GHz]')
        ax[2].set_xlabel('freq [GHz]')
        ax[2].set_ylabel('|z12| [dB]')
        ax[1].legend()
        ax[2].legend()

        # plot phase
        ax[3].plot(freqs, np.angle(Z11), marker='o', label=fr'$\angle$ z11 {which}', mfc='none', lw=0)
        ax[4].plot(freqs, np.angle(Z21), marker='o', label=fr'$\angle$ z12 {which}', mfc='none', lw=0)
        ax[3].set_xlabel('freq [GHz]')
        ax[4].set_ylabel(r'$\angle$ z11 [deg]')
        ax[4].set_xlabel('freq [GHz]')
        ax[4].set_ylabel(r'$\angle$ z11 [deg]')
        ax[3].legend()
        ax[4].legend()

        peaks, _ = find_peaks(20*np.log10(np.abs(Z11)))
        print(f"frequency peaks {which}: \n\t", freqs[peaks])
        # save Z parameters
        # import pickle
        # with open('Zparametes_coarse_mesh.pickle', 'wb') as handle:
        #     pickle.dump(Z, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return ax

    def rom(self, tol=1e-19):
        print('Wfull shape:', self.Wfull.shape)
        U, S, Vt = np.linalg.svd(self.Wfull, full_matrices=False)
        fig,ax = plt.subplot_mosaic([[1]], layout='constrained', figsize=(8,4))
        ax[1].plot(S)
        ax[1].set_yscale('log')
        plt.grid()
        plt.show()

        print('\tDecomp shapes', U.shape, S.shape, Vt.shape)
        # r = len(S[S>tol])
        r = len(S)
        Sr = S[:r]
        Vtr = Vt[:r, :]
        Wr = U[:, :r]
        print('\tTrunc shapes', Wr.shape, Sr.shape, Vtr.shape, r)

        # self._test_reconstruction(W, Wr, Sr, Vtr)
        Rr = Wr.T@self.Ac@Wr
        Rr = (Rr + Rr.T)/2
        # self._check_reduced_model(Wr, Rr, Mr)

        Ard = Rr
        Ard = (Ard + Ard.T)/2
        Brd = Wr.T @ self.Bc

        self.Wr, self.Ard, self.Brd = Wr, Ard, Brd
        return Wr, Ard, Brd

