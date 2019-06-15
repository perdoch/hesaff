#export PS4='+(${BASH_SOURCE}:${LINENO}): ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'
#set -x
__heredoc__="""
TODO: figure out what multibuild expects
"""

function pre_build {
    # Any stuff that you need to do before you start building the wheels
    # Runs in the root directory of this repository.
    :
    echo "todo: define pre_build"
}

function run_tests {
    # Runs tests on installed distribution from an empty directory
    python --version
    echo "todo: run tests"
    #python -c 'import sys; import yourpackage; sys.exit(yourpackage.test())'
}


function setup_venv(){
    #$PYTHON_EXE -m pip install --upgrade pip
    $PYTHON_EXE -m pip install virtualenv 
    $PYTHON_EXE -m virtualenv --python=$PYTHON_EXE venv
    pip install virtualenv
    source $HOME/venv/bin/activate && \
    python --version # just to check && \
    pip install --upgrade pip wheel
}


