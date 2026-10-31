"""Unified native build entry point; invoke on the OS being built for."""
import argparse
from pathlib import Path
import subprocess
import sys
from platform_dispatch import select_platform


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument('--platform', choices=('auto', 'windows', 'linux'), default='auto')
    p.add_argument('--print-command', action='store_true')
    a, remaining = p.parse_known_args(argv)
    try:
        host = select_platform(a.platform)
    except ValueError as error:
        p.error(str(error))
    command = [sys.executable, str(Path(__file__).with_name(f'build_{host}.py')), *remaining]
    if '--help' in remaining or '-h' in remaining:
        print('Unified build: build.py [--platform auto|windows|linux] [--print-command] <helper arguments>\n'
              'Run on the selected OS; no automatic WSL/remote dispatch. Helper options follow:', flush=True)
    if a.print_command:
        print(subprocess.list2cmdline(command))
        return 0
    return subprocess.call(command)


if __name__ == '__main__':
    raise SystemExit(main())
