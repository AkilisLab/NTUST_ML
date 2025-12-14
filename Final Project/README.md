# Taxi Trajectory Destination Prediction (ECML/PKDD 2015, Porto)

Predict final taxi trip destination (latitude, longitude) from partial trajectories and metadata.

## 1. Repository Structure
```
data_understanding.py      # Load raw Kaggle zip/csv data
data_visualization.py      # EDA plots (counts, maps, trajectories)
data_preparation.py        # Build processed tables: train_processed.csv / test_processed.csv
prepare_ml_data.py         # Create ML matrices + prefix features + train/val split
train.py                   # Train baseline models, write metrics + submission_*.csv
results_visualization.py   # Plot comparisons from baseline_metrics.csv
submission_writer.py       # Submission CSV writer (TRIP_ID, LATITUDE, LONGITUDE)
run_all.py                 # Run EDA + preprocessing end-to-end
models/
  baselines/               # Baseline model factories (sklearn)
  evaluation.py            # Metrics (MSE/R2 + Haversine, within-km)
pkdd-15-predict-taxi-service-trajectory-i/  # Kaggle data (zips/csv)
requirements.txt           # Project dependencies

# generated outputs
train_processed.csv, test_processed.csv
baseline_metrics.csv
submission_*.csv
```

## 2. Setup
Create and activate a virtual environment, then install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Data Acquisition
Place the Kaggle competition files in `./pkdd-15-predict-taxi-service-trajectory-i/` (this repo already expects the zipped files):
```
sampleSubmission.csv.zip
train.csv.zip
test.csv.zip
metaData_taxistandsID_name_GPSlocation.csv.zip
```

## 4. Data Understanding & Visualization
Run visualizations (PNG files saved in project root):
```bash
python data_visualization.py
```
Generates:
- `trip_counts_distribution.png`
- `call_type_distribution.png`
- `trajectory_length_distribution.png`
- `start_locations_map.png`
- `trajectories_sample.png`
- `destinations_map.png`

Run the end-to-end EDA + preprocessing workflow:
```bash
python run_all.py
```
This also produces:
- `missing_rate_table.png`
- `correlation_heatmap.png`
- `train_processed.csv` / `test_processed.csv`

## 5. Data Preparation (Processed Tables)
Generate the processed training/test tables used by the ML pipeline:
```bash
python data_preparation.py
```
Outputs:
- `train_processed.csv`
- `test_processed.csv`

Notes:
- Time features and one-hot call-type features are added.
- Raw `POLYLINE` remains in the processed CSVs; prefix features + labels are derived later in `prepare_ml_data.py`.

## 6. Model Training (Baseline Suite)
Train a suite of scikit-learn baselines and write per-model submissions:
```bash
python train.py
```
This will:
- Train/evaluate multiple models (see `models/baselines/`)
- Write `baseline_metrics.csv` (MSE/R² + Haversine metrics + training time)
- Write one submission per model: `submission_<model>.csv`

The default feature builder in `prepare_ml_data.py` uses a fixed prefix length (`PREFIX_LEN = 20`).

## 7. Inference & Submission Generation
`train.py` generates a submission CSV for each baseline model it trains.

Submission schema (written by `submission_writer.py`):
- `TRIP_ID`
- `LATITUDE`
- `LONGITUDE`

## 8. Submission Format Example
```
TRIP_ID,LATITUDE,LONGITUDE
T1,41.146504,-8.611317
T2,42.230000,-8.629454
T10,42.110000,-8.721111
```

## 9. Extending the Pipeline
Potential improvements:
- Time-aware validation (train earlier weeks → validate later weeks).
- Multiple prefix lengths per trajectory (e.g., 5, 10, 20) to expand training data.
- Additional features: average speed, bearing changes, curvature, H3/geohash cells, destination clustering.
- Uncertainty estimation (predict error radius or quantile regression).
- Ensemble: Combine kNN, cluster-centroid classifier, boosted trees, and sequence models.

## 10. Reproducibility Notes
- Prefix duration assumes 15s sampling interval between points.
- Frequency encoding used for high-cardinality fields (`ORIGIN_CALL`, `TAXI_ID`) to control memory.
- One-hot kept for low-cardinality fields (`CALL_TYPE`, `ORIGIN_STAND`).
- No data leakage: destination is derived from the final `POLYLINE` point for training labels only; prefix features are computed from the initial points only.

## 11. Quick Command Summary
```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 2. EDA
python data_visualization.py

# 3. Build processed tables
python data_preparation.py

# 4. Train baselines + write metrics/submissions
python train.py

# 5. Optional: plot model comparisons
python results_visualization.py
```
