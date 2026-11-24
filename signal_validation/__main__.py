import argparse
import json

from .benchmark import MAX_SHIFTS, run


def shift_count(value):
    count = int(value)
    if not 1 <= count <= MAX_SHIFTS:
        raise argparse.ArgumentTypeError(f'Choose between 1 and {MAX_SHIFTS} shifts.')
    return count


def main():
    parser = argparse.ArgumentParser(description='Run the synthetic EEG/fMRI benchmark.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shifts', type=shift_count, default=200)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    print(json.dumps(run(args.seed, args.shifts), indent=2))


if __name__ == '__main__':
    main()
