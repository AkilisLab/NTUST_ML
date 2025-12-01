"""
Run the end-to-end workflow: data load, visualization, and data preparation.

Usage:
  python run_all.py              # default cut_len=10
  python run_all.py 25           # use cut_len=25 for prefixes
"""
import sys

from data_understanding import load_and_preprocess_data
from data_visualization import (
    plot_trip_counts,
    plot_call_type_distribution,
    plot_trajectory_lengths,
    plot_start_locations,
    plot_trajectories_sample,
    plot_destinations_map,
)
from data_preparation import build_training_table, save_training_tables


def main(cut_len: int = 10) -> None:
    print("[1/3] Loading data...")
    train, test, sample, location = load_and_preprocess_data()

    print("[2/3] Generating visualizations...")
    plot_trip_counts(train.copy())
    plot_call_type_distribution(train.copy())
    plot_trajectory_lengths(train.copy())
    plot_start_locations(train.copy())
    plot_trajectories_sample(train.copy())
    plot_destinations_map(train.copy())

    print("[3/3] Preparing training tables...")
    X, y = build_training_table(train.copy(), cut_len=cut_len)
    save_training_tables(X, y)
    print(f"Saved: prep_X.csv (shape {X.shape}), prep_y.csv (shape {y.shape})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            cut_len = int(sys.argv[1])
        except ValueError:
            raise SystemExit("cut_len must be an integer, e.g., 10 or 25")
    else:
        cut_len = 10
    main(cut_len=cut_len)
