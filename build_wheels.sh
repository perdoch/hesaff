#!/bin/bash
__doc__="""
SeeAlso:
    pyproject.toml
"""
#pip wheel -w wheelhouse .
cibuildwheel --config-file pyproject.toml --platform linux --arch x86_64
