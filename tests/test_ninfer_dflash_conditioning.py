"""Compile and run the prepared NInfer policy contract with no CUDA dependency."""
import argparse
import json
import os
from pathlib import Path
import shutil
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] / 'backends/ninfer')
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args()
    source, build = args.source.resolve(), args.build.resolve()
    compiler = shutil.which('cl' if os.name == 'nt' else 'c++')
    if not compiler:
        parser.error('a C++ compiler environment is required')
    ninja = shutil.which('ninja')
    if not ninja and os.name == 'nt':
        vswhere = Path(os.environ['ProgramFiles(x86)']) / 'Microsoft Visual Studio/Installer/vswhere.exe'
        installation = subprocess.check_output([str(vswhere), '-latest', '-products', '*',
            '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property',
            'installationPath'], text=True).strip()
        ninja = str(Path(installation) / 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe')
    if not ninja:
        parser.error('Ninja is required for the focused build audit')
    test = source / 'tests/models/qwen3_5/test_dflash_conditioning.cpp'
    if not test.is_file():
        parser.error('prepare the pinned NInfer source before running this test')
    project = build / 'project'
    project.mkdir(parents=True, exist_ok=True)
    (project / 'CMakeLists.txt').write_text(
        'cmake_minimum_required(VERSION 3.24)\nproject(ninfer_policy LANGUAGES CXX)\n'
        'find_package(Threads REQUIRED)\n'
        'add_executable(ninfer_dflash_conditioning_host "' + test.as_posix() + '" "' +
        (source / 'src/runtime/engine/public_types.cpp').as_posix() + '")\n'
        'target_link_libraries(ninfer_dflash_conditioning_host PRIVATE Threads::Threads)\n'
        'target_compile_features(ninfer_dflash_conditioning_host PRIVATE cxx_std_20)\n'
        'target_include_directories(ninfer_dflash_conditioning_host PRIVATE "' +
        (source / 'include').as_posix() + '" "' + (source / 'src').as_posix() + '")\n',
        encoding='utf-8')
    subprocess.run(['cmake', '-S', str(project), '-B', str(build), '-G', 'Ninja',
                    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_MAKE_PROGRAM=' + ninja,
                    '-DCMAKE_CXX_COMPILER=' + compiler], check=True)
    if os.name == 'nt':
        # Some localized CMake installations misdecode cl's /showIncludes prefix. Derive the
        # actual bytes before building; correct the rule, never fabricate dependency records.
        command = [compiler, '/nologo', '/std:c++20', '/EHsc', '/Zs', '/showIncludes',
                   '/I' + str(source / 'include'), '/I' + str(source / 'src'), str(test)]
        probe = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        line = next(line for line in probe.stdout.splitlines() if b'dflash_conditioning.h' in line)
        path = re.search(rb'[A-Za-z]:[/\\]', line)
        if path is None:
            raise RuntimeError('MSVC include-prefix probe did not report an absolute header path')
        prefix = line[:path.start()].rstrip() + b' '
        rules = build / 'CMakeFiles/rules.ninja'
        original = rules.read_bytes()
        corrected, count = re.subn(rb'(?m)^msvc_deps_prefix = [^\r\n]*',
                                  lambda _: b'msvc_deps_prefix = ' + prefix, original)
        if count != 1:
            raise RuntimeError('expected one generated MSVC dependency-prefix rule')
        if corrected != original:
            rules.write_bytes(corrected)
        (build / 'msvc-prefix-probe.json').write_text(json.dumps(
            {'command': command, 'actual_prefix_hex': prefix.hex(),
             'corrected_generated_rule': corrected != original}, indent=2), encoding='utf-8')
    plan = subprocess.check_output([ninja, '-C', str(build), '-n', '-d', 'explain',
                                   'ninfer_dflash_conditioning_host'], stderr=subprocess.STDOUT, text=True)
    (build / 'policy-plan.log').write_text(plan, encoding='utf-8')
    print(plan)
    if 'Building CUDA object' in plan or plan.count('Building CXX object') > 2:
        raise RuntimeError('policy test expanded beyond its two host compilations')
    subprocess.run([ninja, '-C', str(build), '-j', '8', 'ninfer_dflash_conditioning_host'], check=True)
    executable = build / ('ninfer_dflash_conditioning_host.exe' if os.name == 'nt'
                          else 'ninfer_dflash_conditioning_host')
    subprocess.run([str(executable)], check=True)


if __name__ == '__main__':
    main()
