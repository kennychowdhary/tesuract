from .multiindex import RecursiveHypMultiIndex
from .multiindex import MultiIndex
from .pce import PCEBuilder
from .pce import PCEReg
from .quadrature import QuadGen
import tesuract.preprocessing
import tesuract.utils

# import tesuract.mpce
from .pce_multitarget_regression import (
    RegressionWrapperCV,
    MRegressionWrapperCV,
    MPCEReg,
)
import tesuract.experimental

from pkg_resources import get_distribution

try:
    __version__ = get_distribution("tesuract").version
except:
    print("Install package then run.")
