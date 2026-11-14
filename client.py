"""A hospital client: trains on local data, shares only model weights.

Usage:
    python client.py --hospital-id 1 [--server localhost:8080]
"""
import argparse
import sys
import time

import numpy as np

from netra.config import (BATCH_SIZE, LOCAL_EPOCHS, NUM_CLASSES, SERVER_PORT,
                          USE_CLASS_WEIGHTS, hospital_name)


def make_train_ds(x, y, batch_size):
    """Shuffled, lightly augmented (flips) tf.data pipeline with one-hot labels."""
    import tensorflow as tf

    y1 = tf.keras.utils.to_categorical(y, NUM_CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((x, y1)).shuffle(len(x), seed=0)

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img, label

    return ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(1)


def build_client(hospital_id, x_train, y_train, x_test, y_test):
    import flwr as fl
    import tensorflow as tf

    from netra.data import class_weights
    from netra.model import build_model

    model = build_model(pretrained=False)  # weights come from the server
    weights = class_weights(y_train) if USE_CLASS_WEIGHTS else None

    class HospitalClient(fl.client.NumPyClient):
        started = False  # set once the server has sent us work

        def get_parameters(self, config):
            return model.get_weights()

        def fit(self, parameters, config):
            self.started = True
            model.set_weights(parameters)
            print(f"[Hospital {hospital_id}] Got global model, training on "
                  f"{len(x_train)} local scans...", flush=True)
            hist = model.fit(
                make_train_ds(x_train, y_train, BATCH_SIZE),
                epochs=LOCAL_EPOCHS,
                class_weight=weights,
                verbose=0,
            )
            loss = float(hist.history["loss"][-1])
            acc = float(hist.history["accuracy"][-1])
            print(f"[Hospital {hospital_id}] Local training done. loss={loss:.4f} acc={acc:.4f}. "
                  "Sending weights only.", flush=True)
            return model.get_weights(), len(x_train), {
                "loss": loss, "accuracy": acc, "hospital_id": int(hospital_id),
            }

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            probs = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
            pred = probs.argmax(1)
            y1 = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
            loss = float(tf.keras.losses.categorical_crossentropy(y1, probs).numpy().mean())
            acc = float((pred == y_test).mean())
            cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
            for t, p in zip(y_test, pred):
                cm[t, p] += 1
            print(f"[Hospital {hospital_id}] Evaluated global model. loss={loss:.4f} acc={acc:.4f}",
                  flush=True)
            return loss, len(x_test), {
                "accuracy": acc,
                "hospital_id": int(hospital_id),
                "cm": ",".join(str(v) for v in cm.ravel()),
            }

    return HospitalClient()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hospital-id", type=int, required=True)
    ap.add_argument("--server", default=f"localhost:{SERVER_PORT}",
                    help="host:port of the Flower server")
    args = ap.parse_args()
    if args.hospital_id < 1:
        sys.exit("--hospital-id must be a positive integer")

    from netra.data import load_hospital

    print(f"[Hospital {args.hospital_id}] {hospital_name(args.hospital_id)}: loading local data...",
          flush=True)
    try:
        x_tr, y_tr, x_te, y_te = load_hospital(args.hospital_id)
    except (FileNotFoundError, ValueError, MemoryError) as e:
        sys.exit(f"FATAL: {e}")
    print(f"[Hospital {args.hospital_id}] {len(x_tr)} train / {len(x_te)} test scans. "
          "This data never leaves this process.", flush=True)

    import flwr as fl

    client = build_client(args.hospital_id, x_tr, y_tr, x_te, y_te)
    for attempt in range(1, 31):
        print(f"[Hospital {args.hospital_id}] Connecting to {args.server} (attempt {attempt}) ...",
              flush=True)
        try:
            fl.client.start_client(server_address=args.server, client=client.to_client())
            break
        except Exception as e:  # noqa: BLE001
            if attempt < 30 and "UNAVAILABLE" in str(e) and not client.started:
                time.sleep(5)  # server not up yet; keep trying until it is
                continue
            sys.exit(f"[Hospital {args.hospital_id}] Could not complete training: "
                     f"{str(e).splitlines()[0] if str(e) else type(e).__name__}")


if __name__ == "__main__":
    main()
