#!/usr/bin/env python3
import subprocess
import sys

def get_version():
    try:
        # Similar to what setuptools_scm does under the hood
        return subprocess.check_output(['git', 'describe', '--tags', '--always', '--dirty'], 
                                        encoding='utf-8').strip()
    except Exception:
        return '0.1.0-unknown'

if __name__ == '__main__':
    print(get_version())
