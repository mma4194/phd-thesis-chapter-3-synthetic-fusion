#!/usr/bin/env python3
"""Validate environment and inputs without starting training."""
import sys
from run import main
if __name__=='__main__':
 sys.argv.append('--preflight-only');sys.exit(main())
