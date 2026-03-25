"""Tests for OCCImporter geometry module.

Tests cover:
- STEP file import and mesh generation
- IGES file import and mesh generation
- Direct From_PyOCC transfer (no temp files)
- Splitting workflow
- Multi-format error handling
- Backward-compatible STEPImporter alias
"""

import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from geometry.importers import OCCImporter, STEPImporter


# ============================================================
# Paths to real geometry files in the examples directory
# ============================================================

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

STEP_FILE_CWG = EXAMPLES_DIR / "cwg" / "circular_waveguide.step"
STEP_FILE_RWG = EXAMPLES_DIR / "rwg_step" / "rectangular_waveguide.step"
STEP_FILE_PILLBOX = EXAMPLES_DIR / "pillbox" / "pillbox.step"
IGES_FILE_TESLA = EXAMPLES_DIR / "tesla_step" / "tesla1cell.iges"
STEP_FILE_SPLIT = EXAMPLES_DIR / "rwg_step_split" / "rectangular_waveguide.step"


def _file_available(path: Path) -> bool:
    """Check if a test geometry file exists."""
    return path.exists()


# ============================================================
# Test: STEP import
# ============================================================

class TestSTEPImport:
    """Test basic STEP file import and mesh generation."""

    @pytest.mark.skipif(
        not _file_available(STEP_FILE_CWG),
        reason="Test STEP file not available"
    )
    def test_step_import_creates_mesh(self):
        """Loading a STEP file should produce a valid mesh."""
        geo = OCCImporter(str(STEP_FILE_CWG), unit='mm', maxh=0.05)

        assert geo.geo is not None, "Geometry should be built"
        assert geo.mesh is not None, "Mesh should be generated"
        assert geo.mesh.ne > 0, "Mesh should have elements"

    @pytest.mark.skipif(
        not _file_available(STEP_FILE_RWG),
        reason="Test STEP file not available"
    )
    def test_step_import_deferred_build(self):
        """With auto_build=False, geometry is loaded but not meshed."""
        geo = OCCImporter(str(STEP_FILE_RWG), unit='mm', auto_build=False)

        assert geo._occ_shape is not None, "OCC shape should be loaded"
        assert geo.geo is None, "Geometry should NOT be built yet"
        assert geo.mesh is None, "Mesh should NOT be generated yet"

        # Now finalize
        geo.finalize(maxh=0.05)
        assert geo.mesh is not None, "Mesh should be generated after finalize"


# ============================================================
# Test: IGES import
# ============================================================

class TestIGESImport:
    """Test IGES file import."""

    @pytest.mark.skipif(
        not _file_available(IGES_FILE_TESLA),
        reason="Test IGES file not available"
    )
    def test_iges_import_creates_mesh(self):
        """Loading an IGES file should produce a valid mesh."""
        geo = OCCImporter(str(IGES_FILE_TESLA), unit='mm', maxh=5.0)

        assert geo.geo is not None, "Geometry should be built"
        assert geo.mesh is not None, "Mesh should be generated"
        assert geo.mesh.ne > 0, "Mesh should have elements"


# ============================================================
# Test: BREP-based PythonOCC → netgen transfer
# ============================================================

class TestBREPTransfer:
    """Verify that BREP transfer is used instead of STEP temp files."""

    @pytest.mark.skipif(
        not _file_available(STEP_FILE_CWG),
        reason="Test STEP file not available"
    )
    def test_build_uses_brep_not_step(self):
        """build() should use BREP transfer, not STEP export."""
        geo = OCCImporter(str(STEP_FILE_CWG), unit='mm', auto_build=False)
        
        # The _pyocc_to_netgen method should exist
        assert hasattr(geo, '_pyocc_to_netgen'), \
            "_pyocc_to_netgen should be available for BREP transfer"

        # Build should succeed
        geo.build()
        assert geo.geo is not None

    @pytest.mark.skipif(
        not _file_available(STEP_FILE_CWG),
        reason="Test STEP file not available"
    )
    def test_old_export_method_removed(self):
        """The _export_occ_shape_to_temp method should no longer exist."""
        geo = OCCImporter(str(STEP_FILE_CWG), unit='mm', auto_build=False)
        assert not hasattr(geo, '_export_occ_shape_to_temp'), \
            "_export_occ_shape_to_temp should be removed"


# ============================================================
# Test: Splitting workflow
# ============================================================

class TestSplitting:
    """Test geometry splitting with planes."""

    @pytest.mark.skipif(
        not _file_available(STEP_FILE_SPLIT),
        reason="Test STEP file not available"
    )
    def test_split_creates_multiple_solids(self):
        """Splitting should produce multiple solids."""
        geo = OCCImporter(str(STEP_FILE_SPLIT), unit='mm', auto_build=False)

        # Get the bounding box to determine where to split
        bb = geo.get_bounding_box()
        z_mid = (bb[0][2] + bb[1][2]) / 2  # Split in the middle along Z

        geo.add_split_z(z_mid)
        geo.split()

        # After split, should have multiple solids
        assert geo._is_split, "Geometry should be marked as split"
        assert geo.geo is not None, "Geometry should be rebuilt after split"

    @pytest.mark.skipif(
        not _file_available(STEP_FILE_SPLIT),
        reason="Test STEP file not available"
    )
    def test_split_and_mesh(self):
        """Split geometry should produce a valid mesh."""
        geo = OCCImporter(str(STEP_FILE_SPLIT), unit='mm', auto_build=False)

        bb = geo.get_bounding_box()
        z_mid = (bb[0][2] + bb[1][2]) / 2

        geo.add_split_z(z_mid)
        geo.split()
        geo.generate_mesh(maxh=0.05)

        assert geo.mesh is not None, "Mesh should be generated"
        assert geo.mesh.ne > 0, "Mesh should have elements"


# ============================================================
# Test: Error handling
# ============================================================

class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_unsupported_format_raises_error(self):
        """Unsupported file extension should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            OCCImporter("geometry.stl", unit='mm')

    def test_nonexistent_file_raises_error(self):
        """Non-existent file should raise RuntimeError."""
        with pytest.raises(RuntimeError):
            OCCImporter("nonexistent.step", unit='mm')

    def test_invalid_unit_raises_error(self):
        """Invalid unit should raise ValueError."""
        if not _file_available(STEP_FILE_CWG):
            pytest.skip("Test STEP file not available")
        with pytest.raises(ValueError, match="Unknown unit"):
            OCCImporter(str(STEP_FILE_CWG), unit='inches')


# ============================================================
# Test: Backward compatibility
# ============================================================

class TestBackwardCompatibility:
    """Test that STEPImporter alias works."""

    def test_step_importer_is_occ_importer(self):
        """STEPImporter should be an alias for OCCImporter."""
        assert STEPImporter is OCCImporter

    @pytest.mark.skipif(
        not _file_available(STEP_FILE_CWG),
        reason="Test STEP file not available"
    )
    def test_step_importer_alias_works(self):
        """STEPImporter alias should create a working geometry."""
        geo = STEPImporter(str(STEP_FILE_CWG), unit='mm', auto_build=False)
        assert isinstance(geo, OCCImporter)
        assert geo._occ_shape is not None


# ============================================================
# Test: Format detection
# ============================================================

class TestFormatDetection:
    """Test that file format is correctly detected."""

    def test_step_extensions(self):
        """Both .step and .stp should be detected as 'step'."""
        assert OCCImporter.FORMAT_MAP['.step'] == 'step'
        assert OCCImporter.FORMAT_MAP['.stp'] == 'step'

    def test_iges_extensions(self):
        """Both .iges and .igs should be detected as 'iges'."""
        assert OCCImporter.FORMAT_MAP['.iges'] == 'iges'
        assert OCCImporter.FORMAT_MAP['.igs'] == 'iges'

    def test_brep_extension(self):
        """.brep should be detected as 'brep'."""
        assert OCCImporter.FORMAT_MAP['.brep'] == 'brep'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
