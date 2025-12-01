from warnings import warn
from functools import partial


class WorkInProgress(Warning): ...


TODO = partial(warn, category=WorkInProgress)
