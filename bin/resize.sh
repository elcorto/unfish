#!/bin/sh

# usage: 
#   this.sh <scale> orig/ small/ 
#
# where <scale> is a number between 0 and 1 (or > 1 if you want bigger images
# :) )


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
