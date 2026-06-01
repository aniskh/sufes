"""Package entrypoint for `python -m sufes`.

This module delegates to `sufes.core.main` so the existing CLI is
available when the package is used with `-m`.
"""

from .core import main


if __name__ == "__main__":
    main()
