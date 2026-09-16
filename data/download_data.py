import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route", choices=["synthetic", "alerce", "plasticc", "ztf_bts"], default="synthetic",
        help="Which data route to prepare. 'synthetic' generates a small "
             "fake dataset instantly so you can test the pipeline today. "
             "'alerce' points you to the implemented real-data fetcher.",
    )
    args = parser.parse_args()

    if args.route == "synthetic":
        print("Generating a small synthetic dataset for pipeline smoke-testing...")
        print("Run: python data/generate_synthetic_data.py")
    else:
        print(__doc__)
        print(f"\nSelected route '{args.route}' requires manual download — "
              f"see the instructions printed above.")

    sys.exit(0)


if __name__ == "__main__":
    main()
