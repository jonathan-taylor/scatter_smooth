#!/usr/bin/env python3
import subprocess

def get_version():
    try:
        # Get the latest tag, or fallback to hash if no tag exists
        desc = subprocess.check_output(
            ['git', 'describe', '--tags', '--always', '--dirty'],
            encoding='utf-8'
        ).strip()
        
        # If it's just a hash (no dots), prefix it to make it PEP 440 compliant
        if '.' not in desc:
            return f"0.0.0+g{desc}"
        
        # If it's a tag like v1.0-4-gabc123, change to 1.0.post4+gabc123
        return desc.lstrip('v').replace('-', '.post', 1).replace('-', '+', 1)
        
    except Exception:
        return "0.0.0"

