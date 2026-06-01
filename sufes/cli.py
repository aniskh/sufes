"""Minimal CLI bridge for the `sufes` package.

Right now this module provides a tiny `main()` that delegates to the
existing script `sufes_general.py` so backward compatibility for the
current CLI is preserved while we migrate functionality into modules.
"""
import runpy
import os


def main(argv=None):
    """Run the legacy script in-module. This preserves the current CLI behavior.

    In future iterations we will reimplement argument parsing here and call
    `sufes.core` functions directly.
    """
    # run the top-level script in its own globals so it behaves like direct execution
    script_path = os.path.join(os.path.dirname(__file__), '..', 'sufes_general.py')
    script_path = os.path.abspath(script_path)
    runpy.run_path(script_path, run_name='__main__')


if __name__ == '__main__':
    main()
