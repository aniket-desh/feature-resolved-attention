#!/usr/bin/env python
"""
Script to run FRA visualizations.

Usage:
    python run_visualization.py          # Generate HTML dashboard
    python run_visualization.py --streamlit  # Run Streamlit app (if available)
"""

import argparse
import subprocess
import webbrowser
from pathlib import Path


def run_html_dashboard():
    """Generate and open HTML dashboard."""
    print("Generating FRA HTML Dashboard...")
    
    # Run the visualization script
    subprocess.run(["python", "-m", "fra.single_sample_viz"], cwd="/root/fra")
    
    # Find the latest dashboard file
    results_dir = Path("/root/fra/fra/results")
    dashboards = list(results_dir.glob("fra_dashboard_*.html"))
    
    if dashboards:
        # Get the most recent file
        latest = max(dashboards, key=lambda p: p.stat().st_mtime)
        print(f"\n✅ Dashboard generated: {latest}")
        print(f"\nTo view the dashboard:")
        print(f"  1. Open in browser: file://{latest.absolute()}")
        print(f"  2. Or copy to your local machine and open")
        
        # Try to open in browser (may not work in remote environments)
        try:
            webbrowser.open(f"file://{latest.absolute()}")
        except:
            pass
    else:
        print("❌ No dashboard found. Please check for errors.")


def run_streamlit():
    """Run Streamlit app if available."""
    streamlit_path = Path("/root/fra/fra/streamlit_app.py")
    
    if streamlit_path.exists():
        print("Starting Streamlit app...")
        print("\nStreamlit will run on: http://localhost:8501")
        print("Press Ctrl+C to stop\n")
        
        subprocess.run(["streamlit", "run", str(streamlit_path)], cwd="/root/fra/fra")
    else:
        print("❌ Streamlit app not found at:", streamlit_path)
        print("Please create streamlit_app.py first")


def main():
    parser = argparse.ArgumentParser(description="Run FRA visualizations")
    parser.add_argument(
        "--streamlit", 
        action="store_true",
        help="Run Streamlit app instead of generating HTML dashboard"
    )
    
    args = parser.parse_args()
    
    if args.streamlit:
        run_streamlit()
    else:
        run_html_dashboard()


if __name__ == "__main__":
    main()