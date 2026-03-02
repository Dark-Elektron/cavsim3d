# component_registry.py
"""Component tagging, caching, and compute method infrastructure."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Union
import hashlib
import json
import pickle
from pathlib import Path
from datetime import datetime
import warnings
from pathlib import Path


class ComputeMethod(Enum):
    """
    Method for computing electromagnetic solution.
    
    Attributes
    ----------
    NUMERIC : 
        Full numerical solution using FEM
    ANALYTICAL :
        Closed-form analytical solution (when available)
    SEMI_ANALYTICAL :
        Analytical modes with numerical corrections
    """
    NUMERIC = auto()
    ANALYTICAL = auto()
    SEMI_ANALYTICAL = auto()
    
    def __str__(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class ComponentTag:
    """
    Immutable identifier for a component configuration.
    
    Two components with the same tag are considered identical for solving
    purposes, enabling solution reuse.
    """
    geometry_type: str
    geometry_hash: str
    mesh_hash: str
    transform_hash: str = ""
    version: str = "1.0"
    
    def __str__(self) -> str:
        return f"{self.geometry_type}:{self.geometry_hash[:8]}"
    
    def __repr__(self) -> str:
        return (
            f"ComponentTag({self.geometry_type}, "
            f"geo={self.geometry_hash[:8]}..., "
            f"mesh={self.mesh_hash[:8]}...)"
        )
    
    @property
    def full_hash(self) -> str:
        """Complete hash combining all components."""
        combined = f"{self.geometry_type}:{self.geometry_hash}:{self.mesh_hash}:{self.transform_hash}:{self.version}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def matches(self, other: 'ComponentTag', ignore_mesh: bool = False) -> bool:
        """
        Check if this tag matches another.
        
        Parameters
        ----------
        other : ComponentTag
            Tag to compare with
        ignore_mesh : bool
            If True, only compare geometry (not mesh parameters)
        """
        if self.geometry_type != other.geometry_type:
            return False
        if self.geometry_hash != other.geometry_hash:
            return False
        if not ignore_mesh and self.mesh_hash != other.mesh_hash:
            return False
        return True
    
    @classmethod
    def from_params(
        cls,
        geometry_type: str,
        params: Dict[str, Any],
        mesh_params: Optional[Dict[str, Any]] = None,
        transform: Optional[Dict[str, Any]] = None
    ) -> 'ComponentTag':
        """Create tag from parameter dictionaries."""
        geo_str = json.dumps(params, sort_keys=True, default=str)
        mesh_str = json.dumps(mesh_params or {}, sort_keys=True, default=str)
        transform_str = json.dumps(transform or {}, sort_keys=True, default=str)
        
        return cls(
            geometry_type=geometry_type,
            geometry_hash=hashlib.sha256(geo_str.encode()).hexdigest(),
            mesh_hash=hashlib.sha256(mesh_str.encode()).hexdigest(),
            transform_hash=hashlib.sha256(transform_str.encode()).hexdigest()
        )


class TaggableMixin:
    """
    Mixin that adds tagging capability to geometries.
    
    Subclasses should implement `_get_geometry_params()` and optionally
    `_get_mesh_params()` for proper tag computation.
    """
    
    _tag: Optional[ComponentTag] = None
    _custom_tag: Optional[str] = None
    _compute_method: ComputeMethod = ComputeMethod.NUMERIC
    
    @property
    def tag(self) -> ComponentTag:
        """Unique identifier for this component's configuration."""
        if self._tag is None:
            self._tag = self._compute_tag()
        return self._tag
    
    @property
    def custom_tag(self) -> Optional[str]:
        """User-defined tag for easier identification."""
        return self._custom_tag
    
    @custom_tag.setter
    def custom_tag(self, value: str) -> None:
        self._custom_tag = value
    
    @property
    def compute_method(self) -> ComputeMethod:
        """Method for computing the solution."""
        return self._compute_method
    
    @compute_method.setter
    def compute_method(self, method: Union[ComputeMethod, str]) -> None:
        if isinstance(method, str):
            method = ComputeMethod[method.upper()]
        self._compute_method = method
    
    @property
    def supports_analytical(self) -> bool:
        """Whether this geometry supports analytical solution."""
        return False
    
    def _compute_tag(self) -> ComponentTag:
        """Compute tag from current state."""
        return ComponentTag.from_params(
            geometry_type=self.__class__.__name__,
            params=self._get_geometry_params(),
            mesh_params=self._get_mesh_params()
        )
    
    def _get_geometry_params(self) -> Dict[str, Any]:
        """Override to return geometry parameters for hashing."""
        return {}
    
    def _get_mesh_params(self) -> Dict[str, Any]:
        """Override to return mesh parameters for hashing."""
        return {'maxh': getattr(self, 'maxh', None)}
    
    def invalidate_tag(self) -> None:
        """Invalidate cached tag (call after modifying geometry)."""
        self._tag = None


@dataclass
class CachedSolution:
    """Container for a cached solution."""
    tag: ComponentTag
    solution_data: Any
    compute_method: ComputeMethod
    frequency: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age_seconds(self) -> float:
        return (datetime.now() - self.created_at).total_seconds()


class SolutionCache:
    """
    Cache for computed solutions.
    
    Stores solutions indexed by ComponentTag to avoid recomputation.
    
    Parameters
    ----------
    max_size : int
        Maximum cached solutions (LRU eviction)
    persist_path : Path, optional
        Path to persist cache to disk
    
    Examples
    --------
    >>> cache = SolutionCache(max_size=100)
    >>> 
    >>> # Check and retrieve
    >>> if component.tag in cache:
    ...     solution = cache.get(component.tag)
    ... else:
    ...     solution = solver.solve(component)
    ...     cache.store(component.tag, solution)
    """
    
    def __init__(self, max_size: int = 1000, persist_path: Optional[Path] = None):
        self._cache: Dict[str, CachedSolution] = {}
        self._access_order: List[str] = []
        self._max_size = max_size
        self._persist_path = persist_path
        self._hits = 0
        self._misses = 0
        
        if persist_path and persist_path.exists():
            self._load()
    
    def has(self, tag: ComponentTag) -> bool:
        """Check if solution is cached."""
        return tag.full_hash in self._cache
    
    def get(self, tag: ComponentTag) -> Optional[CachedSolution]:
        """Get cached solution."""
        key = tag.full_hash
        if key in self._cache:
            self._hits += 1
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        
        self._misses += 1
        return None
    
    def store(
        self,
        tag: ComponentTag,
        solution_data: Any,
        compute_method: ComputeMethod = ComputeMethod.NUMERIC,
        frequency: Optional[float] = None,
        **metadata
    ) -> None:
        """Store solution in cache."""
        key = tag.full_hash
        
        # LRU eviction
        while len(self._cache) >= self._max_size and self._access_order:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)
        
        self._cache[key] = CachedSolution(
            tag=tag,
            solution_data=solution_data,
            compute_method=compute_method,
            frequency=frequency,
            metadata=metadata
        )
        self._access_order.append(key)
    
    def find_matching(
        self, 
        tag: ComponentTag, 
        ignore_mesh: bool = False
    ) -> List[CachedSolution]:
        """Find all cached solutions matching the tag."""
        matches = []
        for cached in self._cache.values():
            if cached.tag.matches(tag, ignore_mesh=ignore_mesh):
                matches.append(cached)
        return matches
    
    def invalidate(self, tag: ComponentTag) -> bool:
        """Remove specific solution from cache."""
        key = tag.full_hash
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cached solutions."""
        self._cache.clear()
        self._access_order.clear()
        self._hits = 0
        self._misses = 0
    
    def _load(self) -> None:
        if self._persist_path and self._persist_path.exists():
            try:
                with open(self._persist_path, 'rb') as f:
                    data = pickle.load(f)
                    self._cache = data.get('cache', {})
                    self._access_order = data.get('order', [])
            except Exception as e:
                warnings.warn(f"Failed to load cache: {e}")
    
    def save(self) -> None:
        """Save cache to disk."""
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, 'wb') as f:
                pickle.dump({'cache': self._cache, 'order': self._access_order}, f)
    
    @property
    def size(self) -> int:
        return len(self._cache)
    
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def stats(self) -> Dict[str, Any]:
        return {
            'size': self.size,
            'max_size': self._max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': f"{self.hit_rate:.1%}"
        }
    
    def __contains__(self, tag: ComponentTag) -> bool:
        return self.has(tag)
    
    def __len__(self) -> int:
        return self.size


# Global cache
_global_cache: Optional[SolutionCache] = None

def get_global_cache() -> SolutionCache:
    """Get the global solution cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = SolutionCache()
    return _global_cache

def set_global_cache(cache: SolutionCache) -> None:
    """Set the global solution cache."""
    global _global_cache
    _global_cache = cache