# 4. Z-Parameter Extraction

With the port fields expanded in the normalised modes, every port-mode $m$ carries a
**modal voltage** and a **modal current**[^marks]:

$$
V_m = \int_{\partial\Omega_\text{port}} \mathbf{E}\cdot\mathbf{e}_m\,\mathrm{d}S ,
\qquad
\mathbf{n}\times\mathbf{H}\big|_\text{port} = \sum_m I_m\,\mathbf{e}_m .
$$

The current is the excitation of [§2](variational.md); the voltage is read off the solution. Since the
field coefficients are $j\mathbf{X}$ and $\int \mathbf{N}_i\cdot\mathbf{e}_m\,\mathrm{d}S = b_{i,m}$,
the voltages for unit-current excitations are $j\mathbf{B}^T\mathbf{X}$. After solving the linear
system for all excitations at a given frequency, the impedance matrix is therefore a single
matrix product:

$$
\mathbf{Z}(\omega) = j \, \mathbf{B}^T \mathbf{X}(\omega)
$$

where $\mathbf{X} = [\mathbf{x}_{1,1} \mid \dots \mid \mathbf{x}_{P,m_P}]$ is the matrix of solution
vectors (one column per excitation) and $\mathbf{B}$ is the (real) port basis matrix. Because
the system matrix is symmetric -- complex symmetric with losses -- $\mathbf{Z}$ is symmetric:
the structure is reciprocal[^pozar]. For a lossless structure $\mathbf{Z}$ is purely imaginary.

## References

///Footnotes Go Here///

[^marks]: R. B. Marks and D. F. Williams, "A general waveguide circuit theory," *J. Res.
    Natl. Inst. Stand. Technol.* **97**(5), 533–562 (1992).
    [doi:10.6028/jres.097.024](https://doi.org/10.6028/jres.097.024)
[^pozar]: D. M. Pozar, *Microwave Engineering*, 4th ed. (Wiley, Hoboken, NJ, 2012), ch. 4.

---

**Next:** [5. Reference impedances and S-parameters](s_parameters.md)
