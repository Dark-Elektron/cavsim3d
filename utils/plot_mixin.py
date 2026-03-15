"""PlotMixin — base class providing plot_s / plot_z / plot_eigenvalues for all result objects."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt


class PlotMixin:
    """
    Mixin that adds plot_s(), plot_z(), and plot_eigenvalues() to any result object
    that exposes the following attributes/properties:

        - self.frequencies   : np.ndarray (Hz)
        - self.Z_dict        : dict  e.g. {'1(1)1(1)': array, ...}
        - self.S_dict        : dict  (same keys)
        - self.calculate_resonant_modes() (optional, needed only for plot_eigenvalues)

    Every method accepts an optional `ax` (matplotlib Axes) for overlaying plots
    and returns ``(fig, ax)``.  If `ax` is None a new figure is created.

    Matplotlib styling kwargs (e.g., linewidth, marker, color, linestyle) are
    passed directly to the plot functions.

    Example
    -------
    >>> fig, ax = fds.fom.plot_s(linewidth=2, marker='o', markersize=4)
    >>> fig, ax = fds.fom.rom.plot_s(ax=ax, label='ROM', linestyle='--', color='red')
    """

    # Default plot styles (can be overridden via kwargs)
    _DEFAULT_PLOT_STYLE: Dict = {
        # 'linewidth': 1.5,
    }

    _DEFAULT_MARKER_STYLE: Dict = {
        # 'markersize': 3,
    }

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
    def _merge_style(defaults: Dict, user_kwargs: Dict) -> Dict:
        """Merge default style with user-provided kwargs (user wins)."""
        merged = defaults.copy()
        merged.update(user_kwargs)
        return merged

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
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend: bool = True,
        grid: bool = True,
        grid_alpha: float = 0.3,
        figsize: Tuple[float, float] = (10, 6),
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
            Base label for the legend. If None the class name is used.
            For multiple parameters the parameter key is appended.
        title : str, optional
            Axis title.
        xlabel : str, optional
            X-axis label. Default: 'Frequency (GHz)'.
        ylabel : str, optional
            Y-axis label. Default: auto-generated based on plot_type.
        legend : bool
            Show legend (default True).
        grid : bool
            Show grid (default True).
        grid_alpha : float
            Grid transparency (default 0.3).
        figsize : tuple
            Figure size if creating new figure (default (10, 6)).
        show : bool
            Call ``plt.show()`` after plotting.
        **kwargs
            Matplotlib plot kwargs: linewidth/lw, linestyle/ls, color, 
            marker, markersize/ms, markerfacecolor, markeredgecolor,
            alpha, zorder, etc.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes

        Examples
        --------
        >>> fig, ax = result.plot_s(linewidth=2, color='blue', marker='o', markersize=4)
        >>> fig, ax = result.plot_s(params=['1(1)1(1)'], linestyle='--', alpha=0.7)
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

        fig, ax = self._ensure_ax(ax, figsize=figsize)
        base_label = label or self.__class__.__name__

        # Merge default style with user kwargs
        style = self._merge_style(self._DEFAULT_PLOT_STYLE, kwargs)

        y_label_type = None
        for param in params:
            if param not in s_dict:
                continue
            lbl = f"{base_label} S{param}" if len(params) > 1 else base_label
            y_label_type = self._apply_data(ax, freq_ghz, s_dict[param], plot_type, lbl, style)

        ax.set_xlabel(xlabel or 'Frequency (GHz)')
        ax.set_ylabel(ylabel or f'|S| ({y_label_type})')
        ax.set_title(title or f'{base_label} S-Parameters')
        
        if legend:
            ax.legend()
        if grid:
            ax.grid(True, alpha=grid_alpha)

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
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend: bool = True,
        grid: bool = True,
        grid_alpha: float = 0.3,
        figsize: Tuple[float, float] = (10, 6),
        show: bool = False,
        **kwargs
    ) -> Tuple:
        """
        Plot Z-parameters vs frequency.

        Parameters
        ----------
        params : list of str, optional
            Z-parameter keys to plot, e.g. ``['1(1)1(1)', '1(1)2(1)']``.
            If None, all available parameters are plotted.
        plot_type : str
            One of ``'db'``, ``'mag'``, ``'phase'``, ``'re'``, ``'im'``.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None a new figure is created.
        label : str, optional
            Base label for the legend.
        title : str, optional
            Axis title.
        xlabel : str, optional
            X-axis label. Default: 'Frequency (GHz)'.
        ylabel : str, optional
            Y-axis label. Default: auto-generated based on plot_type.
        legend : bool
            Show legend (default True).
        grid : bool
            Show grid (default True).
        grid_alpha : float
            Grid transparency (default 0.3).
        figsize : tuple
            Figure size if creating new figure (default (10, 6)).
        show : bool
            Call ``plt.show()`` after plotting.
        **kwargs
            Matplotlib plot kwargs: linewidth/lw, linestyle/ls, color,
            marker, markersize/ms, alpha, zorder, etc.

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

        fig, ax = self._ensure_ax(ax, figsize=figsize)
        base_label = label or self.__class__.__name__

        # Merge default style with user kwargs
        style = self._merge_style(self._DEFAULT_PLOT_STYLE, kwargs)

        y_label_type = None
        for param in params:
            if param not in z_dict:
                continue
            lbl = f"{base_label} Z{param}" if len(params) > 1 else base_label
            y_label_type = self._apply_data(ax, freq_ghz, z_dict[param], plot_type, lbl, style)

        ax.set_xlabel(xlabel or 'Frequency (GHz)')
        ax.set_ylabel(ylabel or f'|Z| ({y_label_type})')
        ax.set_title(title or f'{base_label} Z-Parameters')
        
        if legend:
            ax.legend()
        if grid:
            ax.grid(True, alpha=grid_alpha)

        if show:
            plt.show()
        return fig, ax

    def plot_eigenvalues(
        self,
        n_modes: int = 30,
        ax=None,
        label: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        legend: bool = True,
        grid: bool = True,
        grid_alpha: float = 0.3,
        figsize: Tuple[float, float] = (10, 3),
        show: bool = False,
        **kwargs
    ) -> Tuple:
        """
        Plot resonant eigenfrequencies as a stem/rug plot.

        Calls ``self.calculate_resonant_modes()`` or ``self.get_resonant_frequencies()``
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
        xlabel : str, optional
            X-axis label. Default: 'Frequency (GHz)'.
        legend : bool
            Show legend (default True).
        grid : bool
            Show grid (default True).
        grid_alpha : float
            Grid transparency (default 0.3).
        figsize : tuple
            Figure size if creating new figure (default (10, 3)).
        show : bool
            Call ``plt.show()`` after plotting.
        **kwargs
            Matplotlib plot kwargs: marker, markersize/ms, color,
            markerfacecolor, markeredgecolor, alpha, zorder, etc.

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

        if freqs_hz is None and (hasattr(self, 'calculate_resonant_modes') or hasattr(self, 'get_eigenvalues')):
            try:
                if hasattr(self, 'calculate_resonant_modes'):
                    res = self.calculate_resonant_modes()
                    if isinstance(res, tuple):
                        eigs = res[0]
                    elif isinstance(res, dict):
                        eigs = np.concatenate([v[0] for v in res.values()])
                    else:
                        eigs = res
                else:
                    eigs = self.get_eigenvalues()
                if isinstance(eigs, dict):
                    eigs = np.concatenate(list(eigs.values()))
                eigs_pos = eigs[eigs > 0]
                freqs_hz = np.sqrt(eigs_pos) / (2 * np.pi)
            except Exception:
                pass

        if freqs_hz is None:
            raise RuntimeError(
                f"{self.__class__.__name__} does not expose calculate_resonant_modes() "
                "or get_resonant_frequencies()."
            )

        freqs_ghz = np.sort(freqs_hz[:n_modes]) / 1e9
        fig, ax = self._ensure_ax(ax, figsize=figsize)
        base_label = label or self.__class__.__name__

        # Default marker style for eigenvalue plot
        default_eigen_style = {
            'marker': '|',
            'markersize': 14,
            'linestyle': 'none',
        }
        style = self._merge_style(default_eigen_style, kwargs)

        ax.plot(freqs_ghz, np.zeros_like(freqs_ghz), label=base_label, **style)
        
        ax.set_xlabel(xlabel or 'Frequency (GHz)')
        ax.set_yticks([])
        ax.set_title(title or f'{base_label} Eigenfrequencies')
        
        if legend:
            ax.legend()
        if grid:
            ax.grid(True, axis='x', alpha=grid_alpha)

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
        xlabel: Optional[str] = None,
        legend: bool = True,
        grid: bool = True,
        grid_alpha: float = 0.3,
        figsize: Tuple[float, float] = (10, 6),
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
        xlabel : str, optional
            X-axis label. Default: 'Frequency (GHz)'.
        legend : bool
            Show legend (default True).
        grid : bool
            Show grid (default True).
        grid_alpha : float
            Grid transparency (default 0.3).
        figsize : tuple
            Figure size if creating new figure (default (10, 6)).
        show : bool
            Call ``plt.show()`` after plotting.
        **kwargs
            Matplotlib plot kwargs. For 'both', these apply to both curves.
            Use 'iter_kwargs' and 'res_kwargs' dicts within kwargs for
            separate control.

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

        # Extract separate kwargs for iterations and residuals if provided
        iter_kwargs = kwargs.pop('iter_kwargs', {})
        res_kwargs = kwargs.pop('res_kwargs', {})

        # Check for zeros (direct solver)
        verbose = kwargs.pop('verbose', True)
        if np.all(rd.get('iterations', 0) == 0) and verbose:
            print(f"  Note: {base_label} residuals are zero. "
                  f"This is expected when using a direct solver (solver_type='direct').")

        # Choose aggregated or per-excitation data
        if per_excitation:
            iters_2d = rd.get('iterations_per_excitation', rd['iterations'][:, None])
            res_2d = rd.get('residuals_per_excitation', rd['residuals'][:, None])
        else:
            iters_2d = rd['iterations'][:, None]
            res_2d = rd['residuals'][:, None]

        n_exc = iters_2d.shape[1]

        # Default styles with markers
        default_iter_style = {'marker': 'o', 'markersize': 3}
        default_res_style = {'marker': 's', 'markersize': 3}

        if what == 'both':
            if isinstance(ax, (tuple, list)) and len(ax) == 2:
                ax1, ax2 = ax
                fig = ax1.get_figure()
            else:
                fig, ax1 = self._ensure_ax(None, figsize=figsize)
                ax2 = ax1.twinx()

            # Merge styles
            iter_style = self._merge_style(default_iter_style, {**kwargs, **iter_kwargs})
            res_style = self._merge_style(default_res_style, {**kwargs, **res_kwargs})

            for j in range(n_exc):
                alpha = 0.4 if per_excitation and n_exc > 1 else iter_style.get('alpha', 1.0)
                exc_lbl = f'{base_label} exc {j}' if per_excitation and n_exc > 1 else base_label

                iter_style_j = {**iter_style, 'alpha': alpha}
                res_style_j = {**res_style, 'alpha': alpha}

                iter_label = f'{exc_lbl} iterations' if j == 0 or per_excitation else '_nolegend_'
                res_label = f'{exc_lbl} residual' if j == 0 or per_excitation else '_nolegend_'

                ax1.plot(freq_ghz, iters_2d[:, j], label=iter_label, **iter_style_j)
                ax2.semilogy(freq_ghz, res_2d[:, j] + 1e-30, label=res_label, **res_style_j)

            ax1.set_xlabel(xlabel or 'Frequency (GHz)')
            ax1.set_ylabel('GMRES Iterations', color='tab:blue')
            ax2.set_ylabel('Relative Residual', color='tab:red')
            
            suffix = ' (per excitation)' if per_excitation and n_exc > 1 else ' (iter avg, residual min)'
            ax1.set_title(title or f'{base_label} Iterative Solver Convergence{suffix}')
            
            if legend:
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2)
            
            if grid:
                ax1.grid(True, alpha=grid_alpha)

            if show:
                plt.show()
            return fig, (ax1, ax2)

        # Single plot (iterations or residual)
        fig, ax = self._ensure_ax(ax, figsize=figsize)

        if what == 'iterations':
            style = self._merge_style(default_iter_style, kwargs)
        else:
            style = self._merge_style(default_res_style, kwargs)

        for j in range(n_exc):
            alpha = 0.4 if per_excitation and n_exc > 1 else style.get('alpha', 1.0)
            exc_lbl = f'{base_label} exc {j}' if per_excitation and n_exc > 1 else base_label
            style_j = {**style, 'alpha': alpha}

            if what == 'iterations':
                ax.plot(freq_ghz, iters_2d[:, j], label=exc_lbl, **style_j)
                ax.set_ylabel('GMRES Iterations')
            elif what == 'residual':
                ax.semilogy(freq_ghz, res_2d[:, j] + 1e-30, label=exc_lbl, **style_j)
                ax.set_ylabel('Relative Residual')
            else:
                raise ValueError(f"Unknown what='{what}'. Use 'iterations', 'residual', or 'both'.")

        ax.set_xlabel(xlabel or 'Frequency (GHz)')
        suffix = ' (per excitation)' if per_excitation and n_exc > 1 else ''
        ax.set_title(title or f'{base_label} Solver Convergence{suffix}')
        
        if legend:
            ax.legend()
        if grid:
            ax.grid(True, alpha=grid_alpha)

        if show:
            plt.show()
        return fig, ax

    # ------------------------------------------------------------------
    # Comparison helper
    # ------------------------------------------------------------------

    def compare_s(
        self,
        other: 'PlotMixin',
        params: Optional[List[str]] = None,
        plot_type: str = 'db',
        self_label: Optional[str] = None,
        other_label: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 6),
        show: bool = False,
        self_kwargs: Optional[Dict] = None,
        other_kwargs: Optional[Dict] = None,
    ) -> Tuple:
        """
        Compare S-parameters between this result and another.

        Parameters
        ----------
        other : PlotMixin
            Another result object to compare against.
        params : list of str, optional
            Parameters to plot. If None, uses intersection of available params.
        plot_type : str
            Plot type ('db', 'mag', 'phase', 're', 'im').
        self_label : str, optional
            Label for this result.
        other_label : str, optional
            Label for the other result.
        title : str, optional
            Plot title.
        figsize : tuple
            Figure size.
        show : bool
            Show plot immediately.
        self_kwargs : dict, optional
            Matplotlib kwargs for this result's curves.
        other_kwargs : dict, optional
            Matplotlib kwargs for the other result's curves.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes

        Examples
        --------
        >>> fig, ax = fom.compare_s(cst, params=['1(1)1(1)'],
        ...                         self_kwargs={'linewidth': 2},
        ...                         other_kwargs={'linestyle': '--', 'color': 'red'})
        """
        self_kwargs = self_kwargs or {}
        other_kwargs = other_kwargs or {'linestyle': '--'}

        self_label = self_label or self.__class__.__name__
        other_label = other_label or other.__class__.__name__

        # Find common parameters
        if params is None:
            self_params = set(getattr(self, 'S_dict', {}).keys())
            other_params = set(getattr(other, 'S_dict', {}).keys())
            params = list(self_params & other_params - {'frequencies'})

        fig, ax = self.plot_s(
            params=params,
            plot_type=plot_type,
            label=self_label,
            figsize=figsize,
            **self_kwargs
        )

        other.plot_s(
            params=params,
            plot_type=plot_type,
            ax=ax,
            label=other_label,
            **other_kwargs
        )

        if title:
            ax.set_title(title)

        if show:
            plt.show()

        return fig, ax

    def compare_z(
        self,
        other: 'PlotMixin',
        params: Optional[List[str]] = None,
        plot_type: str = 'db',
        self_label: Optional[str] = None,
        other_label: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 6),
        show: bool = False,
        self_kwargs: Optional[Dict] = None,
        other_kwargs: Optional[Dict] = None,
    ) -> Tuple:
        """
        Compare Z-parameters between this result and another.

        Parameters mirror ``compare_s()``.
        """
        self_kwargs = self_kwargs or {}
        other_kwargs = other_kwargs or {'linestyle': '--'}

        self_label = self_label or self.__class__.__name__
        other_label = other_label or other.__class__.__name__

        if params is None:
            self_params = set(getattr(self, 'Z_dict', {}).keys())
            other_params = set(getattr(other, 'Z_dict', {}).keys())
            params = list(self_params & other_params - {'frequencies'})

        fig, ax = self.plot_z(
            params=params,
            plot_type=plot_type,
            label=self_label,
            figsize=figsize,
            **self_kwargs
        )

        other.plot_z(
            params=params,
            plot_type=plot_type,
            ax=ax,
            label=other_label,
            **other_kwargs
        )

        if title:
            ax.set_title(title)

        if show:
            plt.show()

        return fig, ax