##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: check_version.py
#
# Description: checks Python version
#
##############################################################################

import sys
ATLEASTVERSION39 = (sys.version_info >= (3,9))
ATLEASTVERSION38 = (sys.version_info >= (3,8))
ATLEASTVERSION37 = (sys.version_info >= (3,7))
ATLEASTVERSION36 = (sys.version_info >= (3,6))
assert ATLEASTVERSION39
