# Netr.ai: federated learning for diabetic retinopathy screening

Hospitals can't pool retinal scans because patient data is protected by laws like HIPAA, GDPR and India's DPDPA. This project trains one diabetic retinopathy (DR) grading model across several hospitals without any scan leaving its hospital. Only model weights are exchanged, using [Flower](https://flower.ai/).

This is a research demo. It is not a medical device and it has not been clinically validated.

<!-- TODO: add demo GIF (docs/demo.gif) and hosted dashboard link -->

## How it works

```
Hospital 1 \                                                     
Hospital 2  >-- local weights --> Flower server (FedAvg) --> global model --> dashboard
Hospital 3 /  <-- global weights --
```

1. The server starts from an ImageNet-pretrained MobileNetV2 with 5 outputs (DR grades 0 to 4).
2. Each hospital trains it on its own scans for a few epochs, with flip augmentation.
3. Only the updated weights go back. The server averages them, weighted by scan count (FedAvg).
4. Each hospital tests the new global model on its own held-out scans. The server pools the
   confusion matrices into accuracy, quadratic weighted kappa, and sensitivity and specificity for
   referable DR (grade 2 or higher). These are logged every round.
5. The dashboard shows training progress and grades uploaded scans with the latest global model.

The first version of this project ran on four physical laptops, one server and three hospitals.
`run_demo.py` reproduces that on one machine. Each simulated hospital is a separate process that
reads only its own data and talks to the server over a network port.

## Quick start (Windows, Python 3.11)

```bat
setup.bat
.venv\Scripts\python scripts\prepare_data.py --synthetic --hospitals 3 --per-hospital 60
run_demo.bat --clients 3 --rounds 3
```

`setup.bat` creates `.venv` and installs the pinned requirements. The last command starts the
server, the dashboard and three hospitals. Open http://localhost:8501 to watch.

`--synthetic` generates placeholder images so you can try the pipeline right away. For real data
(APTOS 2019, or anything with a CSV of `id_code,diagnosis` plus the images):

```bat
.venv\Scripts\python scripts\prepare_data.py --source <folder> --hospitals 3 --per-hospital 300 --skew 1.0
```

`--skew` gives each hospital a different mix of grades, which is closer to real hospitals.

### Separate machines
On the server laptop run `start-demo.bat` and allow inbound TCP 8080. On each hospital laptop put
`data_zips\hospital_<ID>_data.zip` in place and run `start-client.bat`, then enter the server's IP.

### Data format
`hospital_<N>_data.zip` containing an `images/` folder and `labels.csv` (or `labels.xlsx`) with two
columns: image name and DR grade (0 to 4).

## Project layout

| Path | What it is |
|---|---|
| `netra/` | Shared config, model definition, data loading, metrics, atomic file writes |
| `server/flower_server.py` | FedAvg server, per-round metrics, checkpoints |
| `client.py` | Hospital client |
| `server/dashboard.py` | Streamlit dashboard and scan grading (`NETRA_REPLAY=1` replays a recorded run) |
| `scripts/prepare_data.py` | Splits a dataset into per-hospital zips |
| `examples/` | Five held-out APTOS scans, one per grade, for the dashboard's example buttons |
| `demo/` | A recorded 10-round run and the final model, used when there is no live run |
| `run_demo.py` | One-machine launcher |
| `tests/` | Unit tests |

## Results

APTOS 2019 training images, 3 simulated hospitals with 300 scans each and deliberately different
grade mixes (`--skew 1.0`). Each hospital holds out 20% for testing, so the global model is scored on
180 scans pooled across hospitals. MobileNetV2 from ImageNet weights, 3 local epochs per round,
learning rate 1e-4, frozen BatchNorm layers. The full log is in `demo/training_metrics.json`.

| Federated round | Accuracy | Quadratic kappa | Referable sensitivity | Referable specificity |
|---|---|---|---|---|
| 1 | 71.7% | 0.57 | 30.6% | 99.3% |
| 5 | 85.6% | 0.79 | 58.3% | 98.6% |
| 10 (final) | 83.9% | 0.82 | 63.9% | 98.6% |

Over rounds 6 to 10, accuracy stayed between 83% and 86%, kappa between 0.77 and 0.84, and
referable sensitivity between 50% and 72%.

Always predicting "No DR" scores 65% accuracy on this test split, so kappa and sensitivity are
the numbers to look at, not accuracy.

Things to keep in mind:
- The test set is small: 180 scans, only 36 of them referable. One more missed case moves
  sensitivity by about 3 points, which is why it jumps around between rounds.
- A referable sensitivity of 64% means roughly one in three serious cases is missed. The model
  separates healthy from mild scans well but grades Moderate, Severe and Proliferative poorly (only
  4 of 11 Proliferative scans are graded correctly). It is not usable as a screening tool.
- I picked the training settings (learning rate, no class weights, frozen BatchNorm) by training
  Hospital 2 on its own and checking its held-out split, which is also part of this test set. The
  numbers are therefore a bit optimistic. A cleaner result needs a separate validation split.
- I haven't compared federated training against ordinary centralized training yet.

## Data and credits

Trained and evaluated on the training set of
[APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) (Kaggle).
The scans are not in this repo, apart from five resized example images in `examples/`. They are one
per grade, picked at random from scans held out of training, and shown with their dataset labels.
Check the competition's data rules before reusing them.

## Limitations

- Hospitals are simulated unless you run it on separate machines, and the data is a public dataset.
- It uses plain FedAvg with no TLS, authentication, secure aggregation or differential privacy.
  Weight updates can still leak information, and anyone who can reach the port can join. Don't use
  it with real patient data as it stands.
- Test scans are a random 20% split per hospital, not split by patient, so metrics are optimistic.
- The classifier doesn't check that an upload is a retinal image.

## Tests

```bat
.venv\Scripts\python -m pytest
```

## Stack

Flower 1.12, TensorFlow/Keras 2.17 (MobileNetV2), Streamlit, scikit-learn, pandas.
