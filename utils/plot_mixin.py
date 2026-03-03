"""PlotMixin — base class providing plot_s / plot_z / plot_eigenvalues for all result objects."""

from typing import List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt


class PlotMixin:
    """
    Mixin that adds plot_s(), plot_z(), and plot_eigenvalues() to any result object
    that exposes the following attributes/properties:

        - self.frequencies   : np.ndarray (Hz)
        - self.Z_dict        : dict  e.g. {'1(1)1(1)': array, ...}
        - self.S_dict        : dict  (same keys)
        - self.get_eigenvalues()  (optional, needed only for plot_eigenvalues)

    Every method accepts an optional `ax` (matplotlib Axes) for overlaying plots
    and returns ``(fig, ax)``.  If `ax` is None a new figure is created.

    Example
    -------
    >>> fig, ax = fds.fom.plot_s()
    >>> fig, ax = fds.fom.rom.plot_s(ax=ax, label='ROM')   # overlay
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_freq_ghz(self) -> np.ndarray:
        return getattr(self, 'frequencies', None)

    @staticmethod
    def _ensure_ax(ax=None, figsize=(10, 6)):
        """Return (fig, ax), creating a new figure if ax is None."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        return fig, ax

    @staticmethod
    def _apply_data(ax, freq_ghz, data, plot_type, label, style_kwargs):
        """Plot one parameter series onto ax."""
        if plot_type == 'db':
            y = 20 * np.log10(np.abs(data) + 1e-15)
            ylabel = 'dB'
        elif plot_type == 'mag':
            y = np.abs(data)
            ylabel = 'Magnitude'
        elif plot_type == 'phase':
            y = np.degrees(np.angle(data))
            ylabel = 'Phase (°)'
        elif plot_type == 're':
            y = np.real(data)
            ylabel = 'Real'
        elif plot_type == 'im':
            y = np.imag(data)
            ylabel = 'Imaginary'
        else:
            raise ValueError(f"Unknown plot_type '{plot_type}'. "
                             "Use 'db', 'mag', 'phase', 're', or 'im'.")
        ax.plot(freq_ghz, y, label=label, **style_kwargs)
        return ylabel

    # ------------------------------------------------------------------
    # Public plot methods
    # ------------------------------------------------------------------

    def plot_s(
        self,
        params: Optional[List[str]] = None,
        plot_type: str = 'db',
        ax=None,
        label: Optional[str] = None,
        title: Optional[str] = None,
        show: bool = False,
        **kwargs
    ) -> Tuple:
        """
        Plot S-parameters vs frequency.

        Parameters
        ----------
        params : list of str, optional
            S-parameter keys to plot, e.g. ``['1(1)1(1)', '1(1)2(1)']``.
            If None, all available parameters are plotted.
        plot_type : str
            One of ``'db'``, ``'mag'``, ``'phase'``, ``'re'``, ``'im'``.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None a new figure is created.
        label : str, optional
            Base label for the legend.  If None the class name is used.
            For multiple parameters the parameter key is appended.
        title : str, optional
            Axis title.
        show : bool
            Call ``plt.show()`` after plotting.
        **kwargs
            Passed to ``ax.plot()``.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """

        freqs = self._get_freq_ghz()
        if freqs is None:
            raise RuntimeError("No frequencies available — call solve() first.")
        freq_ghz = freqs / 1e9

        s_dict = getattr(self, 'S_dict', None)
        if s_dict is None:
            raise RuntimeError("No S-parameter data available.")

        if params is None:
            params = [k for k in s_dict.keys() if k != 'frequencies']

        fig, ax = self._ensure_ax(ax)
        base_label = label or self.__class__.__name__

        for param in params:
            if param not in s_dict:
                continue
            lbl = f"{base_label} S{param}" if len(params) > 1 else base_label
            ylabel = self._apply_data(ax, freq_ghz, s_dict[param], plot_type, lbl, kwargs)

        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel(f'|S| ({ylabel})')
        ax.set_title(title or f'{base_label} S-Parameters')
        ax.legend()
        ax.grid(True, alpha=0.3)
        

        if show:
            plt.show()
        return fig, ax

    def plot_z(
        self,
        params: Optional[List[str]] = None,
        plot_type: str = 'db',
        ax=None,
        label: Optional[str] = None,
        title: Optional[str] = None,
        show: bool = False,
        **kwargs
    ) -> Tuple:
        """
        Plot Z-parameters vs frequency.

        Parameters mirror ``plot_s()``.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        

        freqs = self._get_freq_ghz()
        if freqs is None:
            raise RuntimeError("No frequencies available — call solve() first.")
        freq_ghz = freqs / 1e9

        z_dict = getattr(self, 'Z_dict', None)
        if z_dict is None:
            raise RuntimeError("No Z-parameter data available.")

        if params is None:
            params = [k for k in z_dict.keys() if k != 'frequencies']

        fig, ax = self._ensure_ax(ax)
        base_label = label or self.__class__.__name__

        for param in params:
            if param not in z_dict:
                continue
            lbl = f"{base_label} Z{param}" if len(params) > 1 else base_label
            ylabel = self._apply_data(ax, freq_ghz, z_dict[param], plot_type, lbl, kwargs)

        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel(f'|Z| ({ylabel})')
        ax.set_title(title or f'{base_label} Z-Parameters')
        ax.legend()
        ax.grid(True, alpha=0.3)
        

        if show:
            plt.show()
        return fig, ax

    def plot_eigenvalues(
        self,
        n_modes: int = 30,
        ax=None,
        label: Optional[str] = None,
        title: Optional[str] = None,
        marker: str = '|',
        show: bool = False,
        **kwargs
    ) -> Tuple:
        """
        Plot resonant eigenfrequencies as a stem/rug plot.

        Calls ``self.get_eigenvalues()`` or ``self.get_resonant_frequencies()``
        (whichever is available).

        Parameters
        ----------
        n_modes : int
            Maximum number of modes to show.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. A new figure is created if None.
        label : str, optional
            Legend label.
        title : str, optional
            Axis title.
        marker : str
            Marker style passed to ``ax.plot()``.
        show : bool
            Call ``plt.show()`` after plotting.
        **kwargs
            Passed to ``ax.plot()``.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        

        # Try get_resonant_frequencies first, then fall back to get_eigenvalues
        freqs_hz = None
        if hasattr(self, 'get_resonant_frequencies'):
            try:
                freqs_hz = self.get_resonant_frequencies()
            except Exception:
                pass

        if freqs_hz is None and hasattr(self, 'get_eigenvalues'):
            try:
                eigs = self.get_eigenvalues()
                if isinstance(eigs, dict):
                    eigs = np.concatenate(list(eigs.values()))
                eigs_pos = eigs[eigs > 0]
                freqs_hz = np.sqrt(eigs_pos) / (2 * np.pi)
            except Exception:
                pass

        if freqs_hz is None:
            raise RuntimeError(
                f"{self.__class__.__name__} does not expose get_eigenvalues() "
                "or get_resonant_frequencies()."
            )

        freqs_ghz = np.sort(freqs_hz[:n_modes]) / 1e9
        fig, ax = self._ensure_ax(ax, figsize=(10, 3))
        base_label = label or self.__class__.__name__

        ax.plot(freqs_ghz, np.zeros_like(freqs_ghz), marker,
                markersize=14, label=base_label, **kwargs)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_yticks([])
        ax.set_title(title or f'{base_label} Eigenfrequencies')
        ax.legend()
        ax.grid(True, axis='x', alpha=0.3)
        # 

        if show:
            plt.show()
        return fig, ax

    def plot_residual(
        self,
        what: str = 'both',
        per_excitation: bool = False,
        ax=None,
        label: Optional[str] = None,
        title: Optional[str] = None,
        show: bool = False,
        **kwargs
    ) -> Tuple:
        """
        Plot iterative solver convergence data vs frequency.

        Requires that the result object has ``_residual_data`` attribute
        (set automatically when using ``solver_type='iterative'``).

        Parameters
        ----------
        what : {'iterations', 'residual', 'both'}
            - ``'iterations'``: GMRES iterations per frequency
            - ``'residual'``: relative residual ||Ax-b||/||b|| per frequency
            - ``'both'``: dual-axis plot (iterations left, residual right)
        per_excitation : bool
            If False (default), plot aggregated curves across excitations
            (iterations mean, residual minimum).
            If True, plot one line per excitation (semi-transparent).
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None a new figure is created.
            Ignored when ``what='both'`` (creates its own twin axes).
        label : str, optional
            Legend label. If None the class name is used.
        title : str, optional
            Axis title.
        show : bool
            Call ``plt.show()`` after plotting.
        **kwargs
            Passed to ``ax.plot()``.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes (or tuple of axes for 'both')
        """

        rd = getattr(self, '_residual_data', None)
        if rd is None:
            raise RuntimeError(
                "No residual data available. Run solve() with "
                "solver_type='iterative' first."
            )

        freq_ghz = rd['frequencies'] / 1e9
        base_label = label or getattr(self, 'domain', self.__class__.__name__)

        # Check for zeros (direct solver)
        if np.all(rd.get('iterations', 0) == 0):
             # Only print if we are not suppressing output
             if kwargs.get('verbose', True):
                 print(f"  Note: {base_label} residuals are zero. "
                       f"This is expected when using a direct solver (solver_type='direct').")

        # Choose aggregated or per-excitation data
        if per_excitation:
            # (n_freqs, n_excitations)
            iters_2d = rd.get('iterations_per_excitation', rd['iterations'][:, None])
            res_2d = rd.get('residuals_per_excitation', rd['residuals'][:, None])
        else:
            # aggregated: (n_freqs,) → (n_freqs, 1)
            iters_2d = rd['iterations'][:, None]
            res_2d = rd['residuals'][:, None]

        n_exc = iters_2d.shape[1]

        if what == 'both':
            if isinstance(ax, (tuple, list)) and len(ax) == 2:
                ax1, ax2 = ax
                fig = ax1.get_figure()
            else:
                fig, ax1 = self._ensure_ax(None)
                ax2 = ax1.twinx()

            for j in range(n_exc):
                alpha = 0.4 if per_excitation and n_exc > 1 else 1.0
                exc_lbl = f'{base_label} exc {j}' if per_excitation and n_exc > 1 else base_label
                ax1.plot(freq_ghz, iters_2d[:, j], 'o-', markersize=3, alpha=alpha,
                         label=f'{exc_lbl} iterations' if j == 0 or per_excitation else '_nolegend_', **kwargs)
                ax2.semilogy(freq_ghz, res_2d[:, j] + 1e-30, 's-', markersize=3, alpha=alpha,
                             label=f'{exc_lbl} residual' if j == 0 or per_excitation else '_nolegend_', **kwargs)
            ax1.set_xlabel('Frequency (GHz)')
            ax1.set_ylabel('GMRES Iterations', color='tab:blue')
            ax2.set_ylabel('Relative Residual', color='tab:red')
            suffix = ' (per excitation)' if per_excitation and n_exc > 1 else ' (iter avg, residual min)'
            ax1.set_title(title or f'{base_label} Iterative Solver Convergence{suffix}')
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2)
            ax1.grid(True, alpha=0.3)
            
            if show:
                plt.show()
            return fig, (ax1, ax2)

        fig, ax = self._ensure_ax(ax)

        for j in range(n_exc):
            alpha = 0.4 if per_excitation and n_exc > 1 else 1.0
            exc_lbl = f'{base_label} exc {j}' if per_excitation and n_exc > 1 else base_label

            if what == 'iterations':
                ax.plot(freq_ghz, iters_2d[:, j], 'o-', markersize=3, alpha=alpha,
                        label=exc_lbl, **kwargs)
                ax.set_ylabel('GMRES Iterations')
            elif what == 'residual':
                ax.semilogy(freq_ghz, res_2d[:, j] + 1e-30, 's-', markersize=3, alpha=alpha,
                            label=exc_lbl, **kwargs)
                ax.set_ylabel('Relative Residual')
            else:
                raise ValueError(f"Unknown what='{what}'. Use 'iterations', 'residual', or 'both'.")

        ax.set_xlabel('Frequency (GHz)')
        suffix = ' (per excitation)' if per_excitation and n_exc > 1 else ''
        ax.set_title(title or f'{base_label} Solver Convergence{suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        

        if show:
            plt.show()
        return fig, ax

