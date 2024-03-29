# Runs tests for the frontend of the Magnolia compiler.
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import os
import subprocess
import tempfile

magnoliac = 'cabal exec -v0 magnoliac --'

default_test_target = 'tests/frontend/test.mg'
base_path = tempfile.mkdtemp()


class Backend(IntEnum):
    """A backend to emit code in for the Magnolia compiler."""
    Cxx = 0
    JavaScript = 1
    Python = 2


class WriteToFsBehavior(IntEnum):
    """
    A behavior for the compiler to adopt when writing to the filesystem.
    """
    OverwriteTargetFiles = 0  # encountered existing files are overwritten
    WriteIfDoesNotExist = 1  # overwriting files is forbidden


@dataclass
class Config:
    """A configuration with which to call the Magnolia compiler.

    Args:
        backend: the backend for which to compile.
        output_directory: an optional base output directory in which to write
            the files generated by the compilation. When using the compiler in
            build mode, an output directory must be provided.
        import_base_directory: an optional base import directory used for
            inclusion of external modules in the generated source files.
        write_to_fs_behavior: how the compiler should behave with regards to
            overwriting files in the output directory.
    """
    backend: Backend
    output_directory: Optional[str]
    import_base_directory: Optional[str]
    write_to_fs_behavior: WriteToFsBehavior


def _default_config():
    return Config(backend=Backend.Cxx,
                  output_directory=base_path,
                  import_base_directory=None,
                  write_to_fs_behavior=WriteToFsBehavior.WriteIfDoesNotExist)


def mkfile(file_path, content=None):
    """
    Create a file. If content is not provided, dummy content is generated.
    """
    content = 'dummy generated file' if content is None else content
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(content)


def run_compiler(config: Config, target=default_test_target):
    """
    Runs the Magnolia compiler with the specified configuration.
    Returns the completed process, along with its captured stderr and stdout
    outputs.
    """
    command_line = [magnoliac, "build"]
    if config.output_directory is not None:
        command_line += ["-o", config.output_directory]
    if config.import_base_directory is not None:
        command_line += ["-i", config.import_base_directory]
    if config.write_to_fs_behavior == WriteToFsBehavior.OverwriteTargetFiles:
        command_line.append("--allow-overwrite")
    command_line.append(target)
    command_line = [' '.join(command_line)]
    return subprocess.run(command_line, capture_output=True, shell=True)


def test_overwrite_behavior():
    config = _default_config()
    file_content = 'dummy file content'

    def _contains_dummy_file_content(file_path):
        with open(file_path, 'r') as f:
            return file_content == f.read()

    # Write to paths that would be modified by compilation of the default test
    # target.
    gen_path_no_ext, _ = os.path.splitext(
        '{}/{}'.format(config.output_directory, default_test_target))

    hpp_path, cpp_path, other_path = (
        gen_path_no_ext + ext for ext in ['.hpp', '.cpp', '.other'])

    for file_path in [hpp_path, cpp_path, other_path]:
        mkfile(file_path, content=file_content)

    # Attempting to compile without overwriting should fail, and the content
    # of all the files should remain unchanged.
    config.write_to_fs_behavior = WriteToFsBehavior.WriteIfDoesNotExist
    process = run_compiler(config, target=default_test_target)

    assert process.returncode != 0, 'Expected compilation to fail'
    assert b'already exists' in process.stderr, (
        'Expected compilation to throw an "already exists" error.')

    for file_path in [hpp_path, cpp_path, other_path]:
        assert _contains_dummy_file_content(file_path), (
            f'Expected {file_path} to be unmodified by compilation')

    # Attempting to compile with overwriting should succeed, and the content
    # of the hpp and cpp files should have changed, but not the one of the
    # other file.
    config.write_to_fs_behavior = WriteToFsBehavior.OverwriteTargetFiles
    process = run_compiler(config, target=default_test_target)

    assert process.returncode == 0, 'Expected compilation to succeed'
    assert process.stderr == b'', 'Expected stderr to be empty'

    for file_path in [hpp_path, cpp_path]:
        assert not _contains_dummy_file_content(file_path), (
            f'Expected {file_path} to be modified by compilation')

    assert _contains_dummy_file_content(other_path), (
        f'Expected {file_path} to be unmodified by compilation')


# TODO(bchetioui, 22/07/21): expand tests of the compiler interface
