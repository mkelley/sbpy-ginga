# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from importlib.metadata import version as _version, PackageNotFoundError
from ginga.misc.Bunch import Bunch

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass


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


def setup_astrometry():
    spec = Bunch(
        path=os.path.join(p_path, "astrometry.py"),
        module="astrometry",
        klass="Astrometry",
        category="sbpy",
        workspace="dialogs",
    )
    return spec
