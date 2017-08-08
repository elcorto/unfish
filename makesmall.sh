#!/bin/sh

# usage: this.sh orig/ small/

orig=$1
small=$2

for fn in $orig/*; do 
    tgt=$small/$(basename $fn)
    if [ -e $tgt ]; then
        echo "skip: $fn"
    else
        echo "      $fn"
        convert -resize 20% $fn $tgt
    fi
done
