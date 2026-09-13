"""Check that both distribution formats contain the native shader sources."""
import sys
import tarfile
import zipfile
from pathlib import Path


def check(directory):
    directory = Path(directory)
    wheels = list(directory.glob('*.whl'))
    archives = list(directory.glob('*.tar.gz'))
    assert len(wheels) == len(archives) == 1, 'Expected one wheel and one sdist'
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_files = archive.namelist()
    with tarfile.open(archives[0]) as archive:
        sdist_files = archive.getnames()
    for label, files in [('wheel', wheel_files), ('sdist', sdist_files)]:
        shaders = [name for name in files if name.endswith('.metal')]
        assert shaders, f'{label} is missing Metal shaders'
        for required in ('LICENSE', 'NOTICE'):
            assert any(name.endswith('/' + required) or name == required for name in files), (label, required)
        assert any('LICENSES/' in name for name in files), f'{label} is missing upstream licenses'
        for required in ('tests/test_attention_core.py', 'benchmarks/validate_nanovllm.py'):
            assert any(name.endswith('/' + required) for name in files), f'{label} is missing {required}'
        print(label, 'OK:', len(shaders), 'shader source files')


if __name__ == '__main__':
    check(sys.argv[1])
