"""Console and log-file output of cavsim3d (the ``cavsim3d`` logger).

Milestones, warnings and errors are printed to the console; ``verbose`` adds
the details.  A log file (``start_file_log``) records everything.

The logger does not propagate to the root logger, so an application that
configures logging itself sees each message once.  To route cavsim3d's
messages into your own handlers, attach them to ``logging.getLogger('cavsim3d')``.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


# Define logging levels
VERBOSE = logging.DEBUG        # 10 - all details
BASICS = logging.INFO          # 20 - intermediate info (verbose-only on console)
MILESTONE = 25                 # 25 - key milestones (always shown on console)

logging.addLevelName(MILESTONE, "MILESTONE")


class ColorFormatter(logging.Formatter):
    """Console formatter: a symbol per level, in ANSI colours when ``color``."""

    COLORS = {
        logging.ERROR: '\x1b[31m',     # Red
        logging.WARNING: '\x1b[38;5;214m',  # Gold/Amber
        MILESTONE: '\x1b[32m',         # Green for milestones
        logging.INFO: '\x1b[34m',      # Blue (Information/Basics)
        'RUNNING': '\x1b[36m',         # Cyan
        'DONE': '\x1b[32m',            # Green
        'RESET': '\x1b[0m'
    }

    def __init__(self, fmt=None, datefmt=None, color: bool = True):
        super().__init__(fmt, datefmt)
        self.color = color

    def format(self, record):
        if self.color:
            color = self.COLORS.get(record.levelno, self.COLORS['RESET'])
            # "running" and "done" are milestones with their own colours
            if hasattr(record, 'sublevel'):
                if record.sublevel == 'RUNNING':
                    color = self.COLORS['RUNNING']
                elif record.sublevel == 'DONE':
                    color = self.COLORS['DONE']
            reset = self.COLORS['RESET']
        else:
            color = reset = ''

        msg = super().format(record)

        if record.levelno == logging.ERROR:
            return f"{color}❌ ERROR:: {msg}{reset}"
        elif record.levelno == logging.WARNING:
            return f"{color}⚠️  WARNING:: {msg}{reset}"
        elif hasattr(record, 'sublevel'):
            return f"{color}{msg}{reset}"
        elif record.levelno == MILESTONE:
            return f"{color}✅ {msg}{reset}"
        elif record.levelno == logging.INFO:
            return f"{color}INFO:: {msg}{reset}"

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


def _supports_color(stream) -> bool:
    """True if ``stream`` shows ANSI colours: a terminal or a Jupyter kernel.

    ``NO_COLOR`` (any value) turns colours off, ``FORCE_COLOR`` on.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        if stream.isatty():
            return True
    except Exception:
        pass
    # Jupyter's output stream is not a terminal, but renders ANSI colours
    return type(stream).__module__.startswith("ipykernel")


class ConsoleHandler(logging.Handler):
    """Writes each record to the CURRENT ``sys.stdout``.

    Looking the stream up at every record lets ``contextlib.redirect_stdout``
    and test capture see the output.  A character the stream cannot encode
    (an emoji on a cp1252 Windows console) is replaced instead of failing.
    """

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self._color = ColorFormatter('%(message)s', color=True)
        self._plain = ColorFormatter('%(message)s', color=False)

    def emit(self, record):
        try:
            stream = sys.stdout
            if stream is None:
                return
            fmt = self._color if _supports_color(stream) else self._plain
            _write(stream, fmt.format(record) + "\n")
            stream.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


# Initialize logger
logger = logging.getLogger('cavsim3d')
logger.setLevel(VERBOSE)  # Logger always accepts everything; handlers filter
logger.propagate = False

console_handler = ConsoleHandler()
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


def push_verbosity(verbose: Optional[bool]) -> Optional[int]:
    """Set the console verbosity for one operation; ``None`` leaves it as is.

    Returns the previous level for :func:`pop_verbosity` (None if unchanged).
    """
    if verbose is None:
        return None
    previous = console_handler.level
    set_verbosity(verbose)
    return previous


def pop_verbosity(previous: Optional[int]) -> None:
    """Restore the console level returned by :func:`push_verbosity`."""
    if previous is not None:
        console_handler.setLevel(previous)


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


def _write(stream, text: str) -> None:
    """Write ``text``, replacing characters the stream cannot encode."""
    try:
        stream.write(text)
    except UnicodeEncodeError:
        enc = getattr(stream, 'encoding', None) or 'ascii'
        stream.write(text.encode(enc, 'replace').decode(enc, 'replace'))


def echo(*values, sep: str = ' ', end: str = '\n') -> None:
    """``print()`` that never fails on a character the console cannot show.

    Tables and symbols (box drawing, Greek letters) print as they are on a
    UTF-8 stream and with ``?`` in their place on, e.g., a cp1252 pipe.
    """
    stream = sys.stdout
    if stream is None:
        return
    _write(stream, sep.join(str(v) for v in values) + end)
    stream.flush()


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
