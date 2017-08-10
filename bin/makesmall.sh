#!/bin/sh

# usage: this.sh 0.2 orig/ small/

percent=$(echo "$1 * 100" | bc)
orig=$2
small=$3

for fn in $orig/*; do 
    tgt=$small/$(basename $fn)
    if [ -e $tgt ]; then
        echo "skip: $fn"
    else
        echo "      $fn"
        convert -resize ${percent}% $fn $tgt
    fi
done
