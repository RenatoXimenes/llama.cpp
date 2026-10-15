#!/usr/bin/env python3
"""Build, test the packaged server, and publish immutable GitLab packages."""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_URL = ('https://huggingface.co/ggml-org/test-model-stories260K/resolve/'
             '479896ec924af6d40fd419ab8f4d1eb2101de00d/stories260K-f32.gguf')
MODEL_SHA = '270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d'


def run(command, **kwargs):
    print('+ ' + ' '.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def capture(command):
    return run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode('utf-8', 'replace').strip()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def runtime_files(directory):
    return sorted(p for p in directory.iterdir() if p.is_file() and
                  (p.name in ('llama-server', 'llama-server.exe') or
                   p.suffix.lower() == '.dll' or '.so' in p.name))


def build_configuration(build):
    values = {}
    for line in (build / 'CMakeCache.txt').read_text(encoding='utf-8').splitlines():
        if not line or line.startswith(('#', '//')) or '=' not in line:
            continue
        name, value = line.split('=', 1)
        key = name.split(':', 1)[0]
        if key.startswith(('GGML_', 'LLAMA_BUILD_', 'CMAKE_CXX_COMPILER', 'CMAKE_CUDA_COMPILER')):
            values[key] = value
    return values


def copy_cuda_runtime(package):
    toolkit = Path(os.environ['CUDA_PATH'])
    for pattern in ('cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll'):
        matches = sorted((toolkit / 'bin').glob(pattern))
        if not matches:
            raise RuntimeError(f'Missing CUDA runtime: {pattern} in {toolkit / "bin"}')
        for source in matches:
            shutil.copy2(source, package / source.name)
    notices = [p for p in sorted(toolkit.iterdir()) if p.is_file() and
               p.name.upper().startswith(('EULA', 'LICENSE'))]
    if not notices:
        raise RuntimeError(f'CUDA redistribution notices not found in {toolkit}')
    for source in notices:
        shutil.copy2(source, package / ('CUDA-' + source.name))
    # NVIDIA redistributable archives retain component-specific notices.
    for directory in sorted(toolkit.glob('*-archive')):
        if directory.is_dir():
            for source in sorted(directory.iterdir()):
                if source.is_file() and source.name.upper().startswith(('EULA', 'LICENSE')):
                    shutil.copy2(source, package / ('CUDA-' + directory.name + '-' + source.name))


def publish(files, version):
    env = os.environ
    trusted = (env.get('CI_COMMIT_BRANCH') == env.get('CI_DEFAULT_BRANCH') and
               bool(env.get('CI_DEFAULT_BRANCH'))) or env.get('CI_COMMIT_REF_PROTECTED') == 'true'
    if env.get('PUBLISH_PACKAGES') != '1' or not trusted:
        print('Publication skipped; local archives are available in dist/.', flush=True)
        return
    token = env['CI_JOB_TOKEN']
    base = f'{env["CI_API_V4_URL"]}/projects/{env["CI_PROJECT_ID"]}/packages/generic/llama-server/{version}'
    for path in files:
        command = ['curl.exe' if os.name == 'nt' else 'curl', '--fail', '--show-error', '--silent',
                   '--connect-timeout', '20', '--max-time', '1200', '--header', '@-',
                   '--upload-file', str(path), f'{base}/{path.name}']
        ca = env.get('CI_SERVER_TLS_CA_FILE') or env.get('SIMAO_ROOT_CA_CERT')
        if ca:
            command.extend(['--cacert', ca])
        # Keep credentials out of process arguments and console output.
        subprocess.run(command, input=f'JOB-TOKEN: {token}\n'.encode(), check=True)
        print(f'Published: {base}/{path.name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--build-dir', type=Path)
    parser.add_argument('--skip-build', action='store_true', help='Package an existing build; still run regressions')
    parser.add_argument('--model', type=Path, help='Use an existing GGUF instead of downloading the pinned test fixture')
    args = parser.parse_args()
    os.chdir(ROOT)
    system = platform.system().lower()
    if system not in ('linux', 'windows'):
        parser.error('This pipeline packages Linux and Windows only')
    if platform.machine().lower() not in ('x86_64', 'amd64'):
        parser.error('This pipeline targets x86_64; select a separate recipe for other architectures')
    if args.backend == 'cuda' and system != 'windows':
        parser.error('CUDA packaging is currently supported on Windows only')
    jobs = int(os.environ.get('BUILD_JOBS', '4'))
    if jobs < 1:
        parser.error('BUILD_JOBS must be positive')
    name = f'{system}-x64-{args.backend}'
    build = (args.build_dir or ROOT / f'build-gitlab-{name}').resolve()
    output = ROOT / 'dist' / name
    output.mkdir(parents=True, exist_ok=True)
    commit = capture(['git', 'rev-parse', 'HEAD'])
    version = f'{commit[:12]}-{os.environ.get("CI_PIPELINE_ID", "local")}-{os.environ.get("CI_JOB_ID", "build")}'
    info = dict(source_commit=commit, source_dirty=bool(capture(['git', 'status', '--porcelain'])),
                upstream_baseline='b76b13379d3b7d67ab0bc424516bb6fd2522bd8e',
                reused_existing_build=args.skip_build,
                backend=args.backend, platform=name, build_jobs=jobs,
                cmake_version=capture(['cmake', '--version']), python_version=sys.version,
                cuda_architectures=os.environ.get('CUDA_ARCHITECTURES', '86') if args.backend == 'cuda' else None)
    if args.backend == 'cuda':
        info['cuda_version'] = capture(['nvcc', '--version'])
    write_json(output / 'build-info.json', info)
    if not args.skip_build:
        command = ['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
                   '-DLLAMA_BUILD_SERVER=ON', '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF',
                   '-DLLAMA_BUILD_BORINGSSL=ON', '-DBUILD_SHARED_LIBS=ON',
                   '-DGGML_NATIVE=OFF', '-DGGML_BACKEND_DL=ON', '-DGGML_CPU=ON',
                   '-DGGML_CPU_ALL_VARIANTS=ON', '-DGGML_RPC=ON',
                   '-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON',
                   f'-DGGML_CUDA={"ON" if args.backend == "cuda" else "OFF"}']
        if shutil.which('ninja'):
            command.extend(['-G', 'Ninja Multi-Config'])
        elif system == 'windows':
            command.extend(['-A', 'x64'])
        if args.backend == 'cuda':
            command.append(f'-DCMAKE_CUDA_ARCHITECTURES={info["cuda_architectures"]}')
            cccl = os.environ.get('CUDA_CCCL_VERSION', 'v3.4.3')
            if cccl:
                command.append(f'-DGGML_CUDA_CCCL_VERSION={cccl}')
        run(command)
        run(['cmake', '--build', build, '--config', 'Release', '--parallel', str(jobs), '--target', 'llama-server'])
    info['cmake_configuration'] = build_configuration(build)
    write_json(output / 'build-info.json', info)
    binary_name = 'llama-server.exe' if system == 'windows' else 'llama-server'
    bindir = build / 'bin' / 'Release' if (build / 'bin' / 'Release' / binary_name).is_file() else build / 'bin'
    if not (bindir / binary_name).is_file():
        raise RuntimeError(f'Server not found in {bindir}')
    with tempfile.TemporaryDirectory(prefix='llama-package-') as directory:
        package = Path(directory) / 'package'
        package.mkdir()
        for source in runtime_files(bindir):
            shutil.copy2(source, package / source.name)
        if args.backend == 'cuda':
            copy_cuda_runtime(package)
        shutil.copy2(ROOT / 'LICENSE', package / 'LICENSE')
        licenses = re.findall(r'R"=L=\((.*?)\)=L="', (build / 'license.cpp').read_text(encoding='utf-8'), re.S)
        if not licenses:
            raise RuntimeError('CMake dependency licenses not found')
        (package / 'THIRD_PARTY_LICENSES.txt').write_text('\n\n'.join(licenses), encoding='utf-8')
        shutil.copy2(ROOT / 'docs' / 'fork-runtime.md', package / 'FORK-NOTES.md')
        info['server_version'] = capture([package / binary_name, '--version'])
        fixture = args.model.resolve() if args.model else output / 'test-model.gguf'
        if not args.model:
            if not fixture.is_file() or sha256(fixture) != MODEL_SHA:
                urllib.request.urlretrieve(MODEL_URL, fixture)
            if sha256(fixture) != MODEL_SHA:
                raise RuntimeError('Test model checksum mismatch')
        # Test from the staging directory, so missing libraries fail before publication.
        run([sys.executable, ROOT / 'tests/prompt-cache/regression.py', '--binary', package / binary_name,
             '--model', fixture, '--output', output / 'regression'])
        results = json.loads((output / 'regression/results.json').read_text(encoding='utf-8'))
        if len(results) < 7 or not all(row['passed'] for row in results):
            raise RuntimeError('Incomplete regression suite')
        shutil.copy2(output / 'regression/results.json', package / 'regression-results.json')
        manifest = dict(schema_version=1, **info, regression_model_sha256=sha256(fixture),
                        files_sha256={p.name: sha256(p) for p in sorted(package.iterdir())})
        write_json(package / 'build-manifest.json', manifest)
        archive_base = ROOT / 'dist' / f'llama-server-{version}-{name}'
        if system == 'windows':
            archive = archive_base.with_suffix('.zip')
            with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as stream:
                for path in sorted(package.iterdir()):
                    stream.write(path, path.name)
        else:
            archive = archive_base.with_suffix('.tar.gz')
            with tarfile.open(archive, 'w:gz') as stream:
                for path in sorted(package.iterdir()):
                    stream.add(path, arcname=path.name, recursive=False)
        checksum = Path(str(archive) + '.sha256')
        checksum.write_text(f'{sha256(archive)}  {archive.name}\n', encoding='ascii')
        manifest_path = Path(str(archive_base) + '.manifest.json')
        write_json(manifest_path, manifest)
        print(f'Package verified: {archive} ({archive.stat().st_size} bytes)', flush=True)
        if info['source_dirty'] or args.skip_build:
            print('Publication disabled for a modified source tree or reused build.', flush=True)
        else:
            publish([archive, checksum, manifest_path], version)


if __name__ == '__main__':
    main()
