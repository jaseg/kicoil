""" Builds the GUI's coil model in a separate process, so that the GUI stays responsive and can cancel it. """

import logging
import multiprocessing
import os
import threading
import traceback
import warnings

from .geometry import PlanarInductor


def build_model(conn, params):
    """ Build a PlanarInductor from params, where params['shape'] is a function returning the shape. Sends the model
    (or None), the error message (or None) and the log messages and warnings as (tag, text) pairs through conn. Exits
    when the GUI process ends. """
    threading.Thread(target=_exit_with_parent, daemon=True).start()
    messages = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append((record.levelname.lower(),
                                                   f'{record.levelname}: {record.getMessage()}'))
    logging.getLogger('kicoil').addHandler(handler)
    logging.getLogger('kicoil').setLevel(logging.INFO)
    warnings.showwarning = lambda message, *args: messages.append(('warning', str(message)))
    try:
        conn.send((PlanarInductor(**dict(params, shape=params['shape']())), None, messages))
    except ValueError as e:
        conn.send((None, str(e), messages))
    except Exception:
        conn.send((None, f'Unexpected error:\n{traceback.format_exc()}', messages))


def _exit_with_parent():
    multiprocessing.parent_process().join()
    os._exit(1)
