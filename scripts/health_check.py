#!/usr/bin/env python3
"""Quick health check for all configured DeepSafe model services."""

import json
import os
import re
import sys

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package required. Install with: pip install requests")
    sys.exit(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "deepsafe_config.json")


def main():
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    print(f"\n{'Model':<35} {'Type':<8} {'Port':<7} {'Status':<20} {'Loaded'}")
    print("-" * 85)

    for media_type, mt_config in config.get("media_types", {}).items():
        for model_name, health_url in mt_config.get("health_endpoints", {}).items():
            port_match = re.search(r":(\d+)", health_url)
            port = port_match.group(1) if port_match else "?"
            local_url = f"http://localhost:{port}/health"

            try:
                r = requests.get(local_url, timeout=5)
                data = r.json()
                status = data.get("status", "unknown")
                loaded = str(
                    data.get("model_loaded", data.get("model_currently_loaded", "?"))
                )
            except requests.exceptions.ConnectionError:
                status = "unreachable"
                loaded = "-"
            except Exception as e:
                status = f"error: {str(e)[:30]}"
                loaded = "-"

            print(f"{model_name:<35} {media_type:<8} {port:<7} {status:<20} {loaded}")

    print()


if __name__ == "__main__":
    main()
