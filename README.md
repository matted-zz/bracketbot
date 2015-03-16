### Overview

This is a class project from awhile ago (2009), using low-rank matrix
approximation techniques to model basketball score differentials
(using the [pyrsvd](https://code.google.com/p/pyrsvd/) library].  The
code is a mess, but I still run it every year to make new predictions.

### Usage

    $ python parse_season.py
    usage: python parse_season.py cbbgaXX.txt compareXX.txt RANK

### Prediction example

Download the season's data from kenpom (typically `kenpom.com/cbbgaYY.txt`).

    $ python parse_season.py cbbga15.txt ignored 1

The relevant output is:

    RSVD output:
    O-D score (tourn. teams only):
    1   Kentucky             4.472
    2   Gonzaga              3.935
    3   Wisconsin            3.640
    4   Utah                 3.606
    5   Duke                 3.406
    6   Arizona              3.299
    7   Notre Dame           3.250
    8   North Carolina       3.189
    9   Kansas               3.068
    10  Michigan St.         2.970

You can use these summarized strength scores to fill in your bracket
predictions.  You can vary the rank of the matrix decomposition, but
recently I've always been using 1.
