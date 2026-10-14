"""python -m premisecheck: the same command line as `premisecheck`."""
import sys

from .cli import main

sys.exit(main())
