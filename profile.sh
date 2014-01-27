#!/bin/sh
# Globals
export TIMESTAMP=$(date -d "today" +"%Y-%m-%d_%H-%M-%S")
export RAW_SUFFIX=raw.profile.txt
export CLEAN_SUFFIX=clean.profile.txt

# Input
export pyscript_fpath=$1
# Output Stage 1
export line_profile_output=$pyscript_fpath.lprof
# Output Stage 2
export raw_profile=$pyscript_fpath.$TIMESTAMP.$RAW_SUFFIX
export clean_profile=$pyscript_fpath.$TIMESTAMP.$CLEAN_SUFFIX

echo "Profiling $pyscript_fpath"
# Line profile the python code
kernprof.py -l $pyscript_fpath
# Dump the line profile output to a text file
python -m line_profiler $line_profile_output >> $raw_profile
# Clean the line profile output
python pyprofile_clean.py $raw_profile $clean_profile
# Print the cleaned output
cat $clean_profile

remove_profiles()
{
    rm *.profile.txt
}
