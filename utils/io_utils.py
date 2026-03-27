import sys
import os
import hashlib


def get_user_confirmation(message: str, default: bool = True) -> bool:
    """
    Prompt the user for a yes/no confirmation.
    
    Works in both standard terminal and Jupyter environments.
    
    Parameters
    ----------
    message : str
        The message to display to the user.
    default : bool
        The default value if the user just presses Enter.
        
    Returns
    -------
    bool
        True if the user confirmed, False otherwise.
    """
    # Check if we are in an interactive environment
    # In Jupyter, sys.stdin.isatty() might be False but input() still works.
    is_interactive = sys.stdin.isatty()
    
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            is_interactive = True
    except (ImportError, NameError):
        pass

    if not is_interactive:
        # Non-interactive environment - return True with a warning
        print(f"\n[WARNING] Non-interactive environment detected.")
        print(f"[PROMPT] {message}")
        print(f"[ACTION] Proceeding automatically (default={default}).")
        return default

    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        try:
            choice = input(f"\n{message}{suffix} ").lower().strip()
            if not choice:
                return default
            if choice in ('y', 'yes'):
                return True
            if choice in ('n', 'no'):
                return False
            print("Please respond with 'y' or 'n'.")
        except EOFError:
            return default

def strip_timestamps(obj):
    """Recursively remove 'timestamp' keys from dicts/lists."""
    if isinstance(obj, dict):
        return {k: strip_timestamps(v) for k, v in obj.items() if k != 'timestamp'}
    elif isinstance(obj, list):
        return [strip_timestamps(item) for item in obj]
    return obj

def compute_file_hash(filepath):
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def strip_keys(obj, keys_to_strip):
    """Recursively remove specified keys from dicts/lists."""
    if isinstance(obj, dict):
        return {k: strip_keys(v, keys_to_strip)
                for k, v in obj.items() if k not in keys_to_strip}
    elif isinstance(obj, list):
        return [strip_keys(item, keys_to_strip) for item in obj]
    return obj


def deep_diff(obj1, obj2, path=""):
    """Recursively compare two objects and return a list of difference descriptions."""
    diffs = []
    if type(obj1) != type(obj2):
        diffs.append(f"{path}: type changed from {type(obj1).__name__} to {type(obj2).__name__}")
        return diffs

    if isinstance(obj1, dict):
        all_keys = set(obj1.keys()) | set(obj2.keys())
        for key in sorted(all_keys):
            new_path = f"{path}.{key}" if path else key
            if key not in obj1:
                diffs.append(f"{new_path}: added (value: {obj2[key]!r})")
            elif key not in obj2:
                diffs.append(f"{new_path}: removed (was: {obj1[key]!r})")
            else:
                diffs.extend(deep_diff(obj1[key], obj2[key], new_path))
    elif isinstance(obj1, list):
        if len(obj1) != len(obj2):
            diffs.append(f"{path}: list length changed from {len(obj1)} to {len(obj2)}")
        for i in range(min(len(obj1), len(obj2))):
            diffs.extend(deep_diff(obj1[i], obj2[i], f"{path}[{i}]"))
    else:
        if obj1 != obj2:
            diffs.append(f"{path}: {obj1!r} -> {obj2!r}")

    return diffs


def check_source_files(component_sources, geometry_dir="geometry"):
    """
    Check if source files have changed by comparing current file hash
    against the saved hash. Returns a list of difference descriptions.
    """
    diffs = []
    for comp_name, sources in (component_sources or {}).items():
        saved_hash = sources.get('source_hash')
        source_link = sources.get('source_link')
        source_filename = sources.get('source_filename')

        if not saved_hash:
            continue

        # Check internal copy first, then original source_link
        internal_path = (os.path.join(geometry_dir, source_filename)
                         if source_filename else None)

        current_file = None
        if internal_path and os.path.exists(internal_path):
            current_file = internal_path
        elif source_link and os.path.exists(source_link):
            current_file = source_link

        if current_file is None:
            diffs.append(
                f"'{comp_name}': source file not found "
                f"(checked '{internal_path}' and '{source_link}')"
            )
            continue

        current_hash = compute_file_hash(current_file)
        if current_hash != saved_hash:
            diffs.append(
                f"'{comp_name}': source file content has changed ('{current_file}')"
            )

    return diffs
