"""Select the current host OS; this is not a cross-compilation/remote dispatcher."""
import sys


def select_platform(requested='auto', actual=None):
    actual = sys.platform if actual is None else actual
    host = 'windows' if actual == 'win32' else 'linux' if actual.startswith('linux') else None
    if host is None:
        raise ValueError(f'unsupported host OS: {actual}')
    if requested not in ('auto', host):
        raise ValueError(f'--platform {requested} must run on that OS; current host is {host}')
    return host
