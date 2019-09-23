import logging
import sys

FORMAT      = '[Run:AI] [%(levelname)-8s] [%(asctime)s.%(msecs)03d] [%(process)d] [%(filename)-23s:%(lineno)-4d] %(message)s'
DATE_FMT    = '%d-%m-%Y %H:%M:%S'

# FORMATTER
try:
    import coloredlogs
    formatter = coloredlogs.ColoredFormatter(
        fmt             = FORMAT,
        datefmt         = DATE_FMT,
        level_styles    = dict(
            debug       = dict(
                color   = 'blue',
            ),
            info        = dict(
                color   = 'green',
            ),
            warning     = dict(
                color   = 'yellow',
            ),
            error       = dict(
                color   = 'red',
            ),
            critical    = dict(
                color       = 'white',
                background  = 'red',
            ),
        ),
        field_styles = dict(),
    )
except ImportError:
    formatter = logging.Formatter(
        fmt     = FORMAT,
        datefmt = DATE_FMT,
    )

# LOG
log = logging.getLogger("runapy")
log.setLevel(logging.DEBUG)

# STDERR
stderr_handler = logging.StreamHandler(stream=sys.stderr)
stderr_handler.setFormatter(formatter)
log.addHandler(stderr_handler)

debug       = log.debug
info        = log.info
warning     = log.warning
error       = log.error
critical    = log.critical
