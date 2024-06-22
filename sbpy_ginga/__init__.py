# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from warnings import warn
from importlib.metadata import version as _version, PackageNotFoundError
from astropy.utils.exceptions import AstropyWarning

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

try:
    from ginga.misc.Bunch import Bunch
except ImportError:
    warn(AstropyWarning("ginga is not present: sbpy.ginga_plugins will not run."))

    Bunch = None

# path to these plugins
p_path = os.path.split(__file__)[0]


def setup_cometaryenhancements():
    spec = Bunch(
        path=os.path.join(p_path, "cometary_enhancements.py"),
        module="cometary_enhancements",
        klass="CometaryEnhancements",
        category="sbpy",
        workspace="dialogs",
    )
    return spec
