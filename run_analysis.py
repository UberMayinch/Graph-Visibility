#!/usr/bin/env python3
"""
Quick runner script for chaotic systems analysis.
This provides a simple interface to run the main analysis script.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Simple runner that calls the main analysis script."""
    script_dir = Path(__file__).parent
    main_script = script_dir / "analyze_systems.py"
    
    if not main_script.exists():
        print(f"❌ Main analysis script not found: {main_script}")
        sys.exit(1)
    
    # Pass all arguments to the main script
    cmd = [sys.executable, str(main_script)] + sys.argv[1:]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
