osx_stating_helper(){
    echo "THIS IS OXS"
    TAPS="$(brew --repository)/Library/Taps"
    echo "TAPS = $TAPS"
    if [ -e "$TAPS/caskroom/homebrew-cask" -a -e "$TAPS/homebrew/homebrew-cask" ]; then
        rm -rf "$TAPS/caskroom/homebrew-cask"
    fi
    find "$TAPS" -type d -name .git -exec \
            bash -xec '
                cd $(dirname '\''{}'\'') || echo "status: $?"
                git clean -fxd || echo "status: $?"
                sleep 1 || echo "status: $?"
                git status || echo "status: $?"' \; || echo "status: $?"

    brew_cache_cleanup
}

osx_before_install_helper(){
    #if [ ! -d multibuild ]; then
    #    git clone https://github.com/matthew-brett/multibuild.git multibuild
    #fi

    #source multibuild/common_utils.sh
    #source multibuild/osx_utils.sh

    ## Uninstall oclint. See Travis-CI gh-8826
    #brew cask uninstall oclint || true    
    #export CC=clang
    #export CXX=clang++
    #get_macpython_environment $MB_PYTHON_VERSION venv
    #source venv/bin/activate
    #pip install --upgrade pip wheel
    echo "not doing anything for osx prebuild"
}


prebuild_osx_brew_stuff(){
    echo "Running for OSX"
    
    local CACHE_STAGE; (echo "$TRAVIS_BUILD_STAGE_NAME" | grep -qiF "final") || CACHE_STAGE=1

    #after the cache stage, all bottles and Homebrew metadata should be already cached locally
    if [ -n "$CACHE_STAGE" ]; then
        brew update
        generate_ffmpeg_formula
        brew_add_local_bottles
    fi

    #echo 'Installing QT4'
    #brew tap | grep -qxF cartr/qt4 || brew tap cartr/qt4
    #brew tap --list-pinned | grep -qxF cartr/qt4 || brew tap-pin cartr/qt4
    #if [ -n "$CACHE_STAGE" ]; then
    #    brew_install_and_cache_within_time_limit qt@4 || { [ $? -gt 1 ] && return 2 || return 0; }
    #else
    #    brew install qt@4
    #fi

    echo 'Installing FFmpeg'

    if [ -n "$CACHE_STAGE" ]; then
        brew_install_and_cache_within_time_limit ffmpeg_opencv || { [ $? -gt 1 ] && return 2 || return 0; }
    else
        brew install ffmpeg_opencv
    fi

    if [ -n "$CACHE_STAGE" ]; then
        brew_go_bootstrap_mode 0
        return 0
    fi
    
    # Have to install macpython late to avoid conflict with Homebrew Python update
    before_install
}


generate_ffmpeg_formula(){
    local FF="ffmpeg"
    local LFF="ffmpeg_opencv"
    local FF_FORMULA; FF_FORMULA=$(brew formula "$FF")
    local LFF_FORMULA; LFF_FORMULA="$(dirname "$FF_FORMULA")/${LFF}.rb"

    local REGENERATE
    if [ -f "$LFF_FORMULA" ]; then
        local UPSTREAM_VERSION VERSION
        _brew_parse_package_info "$FF" " " UPSTREAM_VERSION _ _
        _brew_parse_package_info "$LFF" " " VERSION _ _   || REGENERATE=1
        #`rebuild` clause is ignored on `brew bottle` and deleted
        # from newly-generated formula on `brew bottle --merge` for some reason
        # so can't compare rebuild numbers
        if [ "$UPSTREAM_VERSION" != "$VERSION" ]; then
            REGENERATE=1
        fi
    else
        REGENERATE=1
    fi
    if [ -n "$REGENERATE" ]; then
        echo "Regenerating custom ffmpeg formula"
        # Bottle block syntax: https://docs.brew.sh/Bottles#bottle-dsl-domain-specific-language
        perl -wpe 'BEGIN {our ($found_blank, $bottle_block);}
            if (/(^class )(Ffmpeg)(\s.*)/) {$_=$1.$2."Opencv".$3."\n"; next;}
            if (!$found_blank && /^$/) {$_.="conflicts_with \"ffmpeg\"\n\n"; $found_blank=1; next;}
            if (!$bottle_block && /^\s*bottle do$/) { $bottle_block=1; next; }
            if ($bottle_block) { if (/^\s*end\s*$/) { $bottle_block=0} elsif (/^\s*sha256\s/) {$_=""} next; }
if (/^\s*depends_on "(x264|x265|xvid|frei0r|rubberband)"$/) {$_=""; next;}
            if (/^\s*--enable-(gpl|libx264|libx265|libxvid|frei0r|librubberband)$/) {$_=""; next;}
            ' <"$FF_FORMULA" >"$LFF_FORMULA"
        diff -u "$FF_FORMULA" "$LFF_FORMULA" || test $? -le 1

        (   cd "$(dirname "$LFF_FORMULA")"
            # This is the official way to add a formula
            # https://docs.brew.sh/Formula-Cookbook#commit
            git add "$(basename "$LFF_FORMULA")"
            git commit -m "add/update custom ffmpeg ${VERSION}"
        )
    fi
}



