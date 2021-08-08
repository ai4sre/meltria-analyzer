#!/usr/bin/env python3

import os

from diag_cause import diag

if __name__ == '__main__':
    # Disable multithreading in numpy.
    # see https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    diag.main()
