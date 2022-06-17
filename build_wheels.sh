#!/bin/bash
__doc__="""
SeeAlso:
    pyproject.toml
"""
# python -m build /project --wheel --outdir=/tmp/cibuildwheel/built_wheel --config-setting=-v
#pip wheel -w wheelhouse .
cibuildwheel --config-file pyproject.toml --platform linux --arch x86_64
