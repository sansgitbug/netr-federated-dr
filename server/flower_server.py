"""Federated server: averages hospital weights and logs metrics for the dashboard.

Usage (from the project root):
    python server/flower_server.py [--rounds 10] [--clients 3] [--fresh]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from netra import config  # noqa: E402
from netra.io import atomic_write_json, save_weights_atomic  # noqa: E402
from netra.model import (  # noqa: E402
    build_model,
    quadratic_weighted_kappa,
    referable_sens_spec,
)


def init_global_model(fresh: bool):
    """Resume from the saved global model if present, else start from ImageNet weights."""
    if config.MODEL_PATH.exists() and not fresh:
        model = build_model(pretrained=False)
        model.load_weights(str(config.MODEL_PATH))
        print(f"Resuming from saved global model: {config.MODEL_PATH}", flush=True)
        return model
    try:
        model = build_model(pretrained=True)
        print("Initialised global model from ImageNet-pretrained MobileNetV2.", flush=True)
    except Exception as e:  # no internet for the weight download
        print(f"WARNING: could not load ImageNet weights ({e}); using random init.", flush=True)
        model = build_model(pretrained=False)
    save_weights_atomic(model, config.MODEL_PATH)
    return model


def make_strategy(model, num_rounds: int, clients_required: int):
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

    state = {
        "status": f"Waiting for {clients_required} hospitals to connect...",
        "current_round": 0,
        "total_rounds": num_rounds,
        "connected_clients": 0,
        "clients_required": clients_required,
        "rounds": [],
    }
    pending_fit = {}  # hospital name -> fit metrics for the round in progress

    def publish():
        try:
            atomic_write_json(config.LOGS_PATH, state)
        except OSError as e:  # a dashboard hiccup must never abort training
            print(f"WARNING: could not update {config.LOGS_PATH.name}: {e}", flush=True)

    class SaveModelStrategy(fl.server.strategy.FedAvg):
        def configure_fit(self, server_round, parameters, client_manager):
            state["connected_clients"] = client_manager.num_available()
            state["current_round"] = server_round
            state["status"] = (f"Round {server_round}/{num_rounds}: waiting for "
                               f"{clients_required} hospitals")
            publish()
            # Blocks until enough hospitals have connected, then samples them.
            instructions = super().configure_fit(server_round, parameters, client_manager)
            state["connected_clients"] = client_manager.num_available()
            state["status"] = f"Round {server_round}/{num_rounds}: hospitals training locally"
            publish()
            return instructions

        def aggregate_fit(self, server_round, results, failures):
            params, metrics = super().aggregate_fit(server_round, results, failures)
            pending_fit.clear()
            for _, fit_res in results:
                m = fit_res.metrics
                name = config.hospital_name(m.get("hospital_id"))
                pending_fit[name] = {"train_loss": m.get("loss"), "train_accuracy": m.get("accuracy"),
                                     "examples": fit_res.num_examples}
                print(f"[Round {server_round}] {name}: trained on {fit_res.num_examples} scans, "
                      f"local acc={m.get('accuracy', float('nan')):.4f}", flush=True)
            if failures:
                print(f"[Round {server_round}] {len(failures)} hospital(s) failed to report", flush=True)
            if params is not None:
                model.set_weights(parameters_to_ndarrays(params))
                save_weights_atomic(model, config.MODEL_PATH)
                save_weights_atomic(
                    model, config.CHECKPOINT_DIR / f"round_{server_round:03d}.weights.h5"
                )
                print(f"[Round {server_round}] Aggregated weights saved as the new global model.",
                      flush=True)
            state["status"] = f"Round {server_round}/{num_rounds}: evaluating global model"
            publish()
            return params, metrics

        def aggregate_evaluate(self, server_round, results, failures):
            if not results:
                print(f"[Round {server_round}] No evaluation results.", flush=True)
                return None, {}
            k = config.NUM_CLASSES
            total_cm = np.zeros((k, k))
            n_total, loss_sum, acc_sum = 0, 0.0, 0.0
            per_hospital = {}
            for _, ev in results:
                m = ev.metrics
                name = config.hospital_name(m.get("hospital_id"))
                n_total += ev.num_examples
                loss_sum += ev.num_examples * ev.loss
                acc_sum += ev.num_examples * m.get("accuracy", 0.0)
                cm = np.array([int(v) for v in str(m.get("cm", "")).split(",") if v != ""], dtype=float)
                if cm.size == k * k:
                    total_cm += cm.reshape(k, k)
                    h_kappa = quadratic_weighted_kappa(cm.reshape(k, k))
                else:
                    h_kappa = None
                per_hospital[name] = {
                    **pending_fit.get(name, {}),
                    "loss": ev.loss,
                    "accuracy": m.get("accuracy", 0.0),
                    "kappa": h_kappa,
                    "test_examples": ev.num_examples,
                }
            loss = loss_sum / n_total
            acc = acc_sum / n_total
            kappa = quadratic_weighted_kappa(total_cm)
            sens, spec = referable_sens_spec(total_cm)
            print(f"[Round {server_round}] GLOBAL accuracy={acc:.4f} kappa={kappa:.4f} "
                  f"referable sens={sens:.3f} spec={spec:.3f}", flush=True)
            state["rounds"].append({
                "round": server_round,
                "global_accuracy": acc,
                "global_loss": loss,
                "kappa": kappa,
                "referable_sensitivity": sens,
                "referable_specificity": spec,
                "confusion_matrix": total_cm.astype(int).tolist(),
                "client_metrics": per_hospital,
            })
            state["status"] = (
                "Training complete" if server_round == num_rounds
                else f"Round {server_round}/{num_rounds} complete"
            )
            publish()
            return loss, {"accuracy": acc}

    publish()
    return SaveModelStrategy(
        initial_parameters=ndarrays_to_parameters(model.get_weights()),
        min_fit_clients=clients_required,
        min_evaluate_clients=clients_required,
        min_available_clients=clients_required,
    ), state, publish


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=config.NUM_ROUNDS)
    ap.add_argument("--clients", type=int, default=config.CLIENTS_REQUIRED)
    ap.add_argument("--fresh", action="store_true",
                    help="discard any saved global model and start again from ImageNet weights")
    ap.add_argument("--host", default=config.SERVER_HOST)
    ap.add_argument("--port", type=int, default=config.SERVER_PORT)
    args = ap.parse_args()

    import flwr as fl

    model = init_global_model(args.fresh)
    strategy, state, publish = make_strategy(model, args.rounds, args.clients)

    print(f"Flower server on {args.host}:{args.port}, waiting for {args.clients} hospitals, "
          f"{args.rounds} rounds.", flush=True)
    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    if state["status"] != "Training complete":
        state["status"] = "Stopped before all rounds finished"
        publish()
    print("Server finished.", flush=True)


if __name__ == "__main__":
    main()
