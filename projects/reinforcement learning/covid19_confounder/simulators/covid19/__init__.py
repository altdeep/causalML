"""COVID-19 simulator initialization."""

from simulators.covid19.simulator import (
    Config,
    dynamics,
    Intervention,
    simulate,
    State,
)
from simulators.covid19.experiments import COVID19RCT
from simulators.covid19.environments import *

SUPPORTS_CAUSAL_GRAPHS = True
