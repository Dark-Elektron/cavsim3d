import sys
import os

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
