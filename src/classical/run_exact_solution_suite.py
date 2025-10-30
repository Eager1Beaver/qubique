# src/classical/run_exact_solution_suite.py
import argparse, json, os, subprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="JSON with list of config paths")
    ap.add_argument("--python", default="python", help="Python interpreter")
    args = ap.parse_args()

    with open(args.suite) as f:
        suite = json.load(f)
    for cfg in suite["configs"]:
        print(f"== Running {cfg} ==")
        subprocess.run([args.python, "src/classical/exact_solution.py", "--config", cfg], check=True)

if __name__ == "__main__":
    main()
