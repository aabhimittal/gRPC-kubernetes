#!/usr/bin/env python3
"""Static checks over k8s/: catch the manifest mistakes that only show up as a
bad rollout — a probe pointing at the wrong port, a grace period shorter than
the drain the server actually performs, or a missing disruption budget.
"""
import glob
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED_KINDS = {
    "Namespace", "ConfigMap", "Deployment", "Service",
    "HorizontalPodAutoscaler", "PodDisruptionBudget",
}


def main() -> int:
    docs, kinds, errors = [], set(), []
    for path in sorted(glob.glob(os.path.join(ROOT, "k8s", "*.yaml"))):
        for doc in yaml.safe_load_all(open(path)):
            if doc:
                docs.append((os.path.relpath(path, ROOT), doc))
                kinds.add(doc["kind"])

    missing = REQUIRED_KINDS - kinds
    if missing:
        errors.append(f"missing manifest kinds: {sorted(missing)}")

    config = next((d for _, d in docs if d["kind"] == "ConfigMap"), {}).get("data", {})
    drain_ms = int(config.get("PRESTOP_SLEEP_MS", 0)) + int(config.get("GRACE_MS", 0))

    for path, doc in docs:
        if doc["kind"] != "Deployment":
            continue
        spec = doc["spec"]["template"]["spec"]
        container = spec["containers"][0]
        ports = {p["name"]: p["containerPort"] for p in container["ports"]}
        name = doc["metadata"]["name"]

        if "grpc" not in ports or "metrics" not in ports:
            errors.append(f"{path}: {name} must expose both grpc and metrics ports")
        if str(ports.get("metrics")) != config.get("METRICS_PORT"):
            errors.append(
                f"{path}: {name} metrics port {ports.get('metrics')} != "
                f"ConfigMap METRICS_PORT {config.get('METRICS_PORT')}"
            )
        if str(ports.get("grpc")) != config.get("PORT"):
            errors.append(
                f"{path}: {name} grpc port {ports.get('grpc')} != "
                f"ConfigMap PORT {config.get('PORT')}"
            )
        for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
            if probe not in container:
                errors.append(f"{path}: {name} has no {probe}")
            elif container[probe].get("grpc", {}).get("port") != ports.get("grpc"):
                errors.append(f"{path}: {name} {probe} does not target the grpc port")

        grace_s = spec.get("terminationGracePeriodSeconds", 30)
        if grace_s * 1000 <= drain_ms:
            errors.append(
                f"{path}: {name} terminationGracePeriodSeconds={grace_s} does not "
                f"cover the server's own drain of {drain_ms}ms — the kubelet would "
                f"SIGKILL mid-shutdown"
            )
        if doc["spec"].get("strategy", {}).get("rollingUpdate", {}).get("maxUnavailable") != 0:
            errors.append(f"{path}: {name} rollout may drop capacity (maxUnavailable != 0)")

    for err in errors:
        print(f"ERROR: {err}", file=sys.stderr)
    print(f"checked {len(docs)} manifests, {len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
