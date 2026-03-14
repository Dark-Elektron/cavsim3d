import logging
import sys

# Define logging levels
BASICS = logging.INFO
VERBOSE = logging.DEBUG

class ColorFormatter(logging.Formatter):
    """Custom logging formatter that adds ANSI colors."""
    
    COLORS = {
        logging.ERROR: '\x1b[31m',   # Red
        logging.WARNING: '\x1b[33m', # Yellow
        logging.INFO: '\x1b[34m',    # Blue (Information/Basics)
        'RUNNING': '\x1b[36m',       # Cyan
        'DONE': '\x1b[32m',          # Green
        'RESET': '\x1b[0m'
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.COLORS['RESET'])
        
        # Special handling for "running" and "done" which might map to INFO but need different colors
        if hasattr(record, 'sublevel'):
            if record.sublevel == 'RUNNING':
                color = self.COLORS['RUNNING']
            elif record.sublevel == 'DONE':
                color = self.COLORS['DONE']

        msg = super().format(record)
        
        if record.levelno == logging.ERROR:
            return f"{color}ERROR:: {msg}{self.COLORS['RESET']}"
        elif record.levelno == logging.WARNING:
            return f"{color}WARNING:: {msg}{self.COLORS['RESET']}"
        elif hasattr(record, 'sublevel'):
             return f"{color}{msg}{self.COLORS['RESET']}"
        elif record.levelno == logging.INFO:
            return f"{color}INFO:: {msg}{self.COLORS['RESET']}"
        
        return msg

# Initialize logger
logger = logging.getLogger('cavsim3d')
logger.setLevel(VERBOSE)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColorFormatter('%(message)s'))
logger.addHandler(console_handler)

def set_verbosity(verbose: bool):
    """Set the global verbosity level."""
    if verbose:
        logger.setLevel(VERBOSE)
    else:
        logger.setLevel(BASICS)

def error(msg):
    logger.error(msg)

def warning(msg):
    logger.warning(msg)

def running(msg):
    logger.log(BASICS, msg, extra={'sublevel': 'RUNNING'})

def info(msg):
    logger.info(msg)

def done(msg):
    logger.log(BASICS, msg, extra={'sublevel': 'DONE'})

def debug(msg):
    logger.debug(msg)