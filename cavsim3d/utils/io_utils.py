
import sys
import os
import hashlib
from typing import Optional
from IPython import get_ipython



def is_interactive() -> bool:
    """True in a terminal session or a Jupyter/IPython kernel (someone can answer)."""
    try:
        if get_ipython() is not None:
            return True
    except (ImportError, NameError):
        pass
    return sys.stdin.isatty()


def _no_input_errors() -> tuple:
    """Exceptions ``input()`` raises when nobody can answer.

    EOFError: stdin is closed; OSError: stdin is captured (pytest);
    StdinNotImplementedError: a kernel run without a frontend (nbconvert,
    papermill, a docs build).
    """
    errors = [EOFError, OSError]
    try:
        from IPython.core.error import StdinNotImplementedError
        errors.append(StdinNotImplementedError)
    except ImportError:
        pass
    return tuple(errors)


def _ask(prompt: str) -> Optional[str]:
    """``input(prompt)``, or None when no answer is possible."""
    if not is_interactive():
        return None
    try:
        return input(prompt)
    except _no_input_errors():
        return None


def get_user_confirmation(message: str, default: bool = True) -> bool:
    """
    Prompt the user for a yes/no confirmation.

    Works in both standard terminal and Jupyter environments.  When nobody
    can answer (a script without a terminal, a notebook run by nbconvert or
    papermill, closed stdin), the answer is no: a destructive action is never
    taken on a default.

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
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        choice = _ask(f"\n{message}{suffix} ")
        if choice is None:
            # Callers expose force=True for scripts that really mean it.
            print("\n[WARNING] No answer possible (non-interactive session).")
            print(f"[PROMPT] {message}")
            print("[ACTION] Not confirmed -- nothing was changed. "
                  "Pass force=True to proceed without a prompt.")
            return False
        choice = choice.lower().strip()
        if not choice:
            return default
        if choice in ('y', 'yes'):
            return True
        if choice in ('n', 'no'):
            return False
        print("Please respond with 'y' or 'n'.")


def ask_choice(message: str, choices, default: str) -> Optional[str]:
    """Ask for one of ``choices``; Enter picks ``default``.

    Returns None when nobody can answer, so the caller keeps things as they
    are.
    """
    choices = [str(c) for c in choices]
    while True:
        choice = _ask(f"{message} [{'/'.join(choices)}]: ")
        if choice is None:
            return None
        choice = choice.strip() or default
        if choice in choices:
            return choice
        print(f"Please answer one of {choices}.")

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
