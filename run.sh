#!/bin/sh
# Start the page and open it in the browser. Ctrl+C stops it.
cd "$(dirname "$0")" || exit 1
exec python3 -I app/ripple_studio.py --open "$@"
