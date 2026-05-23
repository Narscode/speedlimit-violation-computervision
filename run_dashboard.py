#!/usr/bin/env python3
"""Launcher for the SpeedGuard Vision Interactive Dashboard."""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

PORT = 8000
DIRECTORY = Path(__file__).resolve().parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

def main():
    os.chdir(DIRECTORY)
    
    # Check if dashboard files exist
    dashboard_dir = DIRECTORY / "dashboard"
    if not (dashboard_dir / "index.html").exists():
        print(f"Error: Dashboard files not found in {dashboard_dir}", file=sys.stderr)
        sys.exit(1)

    print("==================================================================")
    print("      🚀 SpeedGuard Vision Interactive Dashboard Launcher 🚀      ")
    print("==================================================================")
    print(f"Starting local server at: http://localhost:{PORT}/dashboard/")
    print("KeyboardInterrupt (Ctrl+C) to stop server.\n")

    # Start the TCP socket server
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            # Auto-open browser in a new tab
            webbrowser.open_new_tab(f"http://localhost:{PORT}/dashboard/")
            
            # Start servicing requests
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Local server terminated. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Server failure: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
