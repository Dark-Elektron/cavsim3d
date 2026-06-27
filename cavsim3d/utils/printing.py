
import logging
import sys
from pathlib import Path


# Define logging levels
VERBOSE = logging.DEBUG        # 10 - all details
BASICS = logging.INFO          # 20 - intermediate info (verbose-only on console)
MILESTONE = 25                 # 25 - key milestones (always shown on console)

logging.addLevelName(MILESTONE, "MILESTONE")

class ColorFormatter(logging.Formatter):
    """Custom logging formatter that adds ANSI colors."""

    COLORS = {
        logging.ERROR: '\x1b[31m',     # Red
        logging.WARNING: '\x1b[38;5;214m',  # Gold/Amber
        MILESTONE: '\x1b[32m',         # Green for milestones
        logging.INFO: '\x1b[34m',      # Blue (Information/Basics)
        'RUNNING': '\x1b[36m',         # Cyan
        'DONE': '\x1b[32m',            # Green
        'RESET': '\x1b[0m'
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.COLORS['RESET'])

        # Special handling for "running" and "done" which might map to MILESTONE but need different colors
        if hasattr(record, 'sublevel'):
            if record.sublevel == 'RUNNING':
                color = self.COLORS['RUNNING']
            elif record.sublevel == 'DONE':
                color = self.COLORS['DONE']

        msg = super().format(record)

        if record.levelno == logging.ERROR:
            return f"{color}\u274c ERROR:: {msg}{self.COLORS['RESET']}"
        elif record.levelno == logging.WARNING:
            return f"{color}\u26a0\ufe0f  WARNING:: {msg}{self.COLORS['RESET']}"
        elif hasattr(record, 'sublevel'):
             return f"{color}{msg}{self.COLORS['RESET']}"
        elif record.levelno == MILESTONE:
            return f"{color}\u2705 {msg}{self.COLORS['RESET']}"
        elif record.levelno == logging.INFO:
            return f"{color}INFO:: {msg}{self.COLORS['RESET']}"

        return msg


class PlainFormatter(logging.Formatter):
    """Plain text formatter for log files (no ANSI colors, with timestamps)."""

    LEVEL_NAMES = {
        logging.DEBUG: 'DEBUG',
        logging.INFO: 'INFO',
        logging.WARNING: 'WARNING',
        logging.ERROR: 'ERROR',
        MILESTONE: 'MILESTONE',
    }

    def format(self, record):
        level = self.LEVEL_NAMES.get(record.levelno, f'LEVEL{record.levelno}')
        if hasattr(record, 'sublevel'):
            level = record.sublevel
        msg = super().format(record)
        timestamp = self.formatTime(record, '%Y-%m-%d %H:%M:%S')
        return f"{timestamp} [{level}] {msg}"


# Initialize logger
logger = logging.getLogger('cavsim3d')
logger.setLevel(VERBOSE)  # Logger always accepts everything; handlers filter

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColorFormatter('%(message)s'))
console_handler.setLevel(MILESTONE)  # Default: show only milestones+ on console
logger.addHandler(console_handler)

# Track active file handlers
_active_file_handlers = []


def set_verbosity(verbose: bool):
    """Set the console verbosity level.

    Parameters
    ----------
    verbose : bool
        If True, console shows all output (DEBUG level).
        If False, console shows only milestones, warnings, and errors.
    """
    if verbose:
        console_handler.setLevel(VERBOSE)
    else:
        console_handler.setLevel(MILESTONE)


def start_file_log(log_path: Path) -> logging.FileHandler:
    """Attach a file handler that captures all output (VERBOSE level).

    Parameters
    ----------
    log_path : Path
        Path to the log file. Parent directories are created if needed.

    Returns
    -------
    logging.FileHandler
        The handler reference (pass to stop_file_log to detach).
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(str(log_path), mode='w', encoding='utf-8')
    handler.setLevel(VERBOSE)
    handler.setFormatter(PlainFormatter('%(message)s'))
    logger.addHandler(handler)
    _active_file_handlers.append(handler)
    return handler


def stop_file_log(handler: logging.FileHandler):
    """Detach and close a file handler.

    Parameters
    ----------
    handler : logging.FileHandler
        The handler returned by start_file_log.
    """
    if handler in _active_file_handlers:
        _active_file_handlers.remove(handler)
    logger.removeHandler(handler)
    handler.flush()
    handler.close()


def close_all_file_logs():
    """Detach and close every active file handler.

    Useful on Windows before deleting a project directory: an open log file
    handle keeps the file locked, which makes ``shutil.rmtree`` fail.
    """
    for handler in list(_active_file_handlers):
        try:
            logger.removeHandler(handler)
            handler.flush()
            handler.close()
        except Exception:
            pass
        finally:
            if handler in _active_file_handlers:
                _active_file_handlers.remove(handler)


def read_log(log_path: Path) -> str:
    """Read and return the contents of a log file.

    Parameters
    ----------
    log_path : Path
        Path to the log file.

    Returns
    -------
    str
        Log file contents, or a message if not found.
    """
    log_path = Path(log_path)
    if log_path.exists():
        return log_path.read_text(encoding='utf-8')
    return f"No log file found at: {log_path}"


def error(msg):
    logger.error(msg)

def warning(msg):
    logger.warning(msg)

def running(msg):
    logger.log(MILESTONE, msg, extra={'sublevel': 'RUNNING'})

def info(msg):
    logger.info(msg)

def done(msg):
    logger.log(MILESTONE, msg, extra={'sublevel': 'DONE'})

def debug(msg):
    logger.debug(msg)

def milestone(msg):
    logger.log(MILESTONE, msg)
