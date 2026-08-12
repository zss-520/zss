from model_resource_policy import apply_resource_gate


def _rows(*models):
    return [
        {"model": model, "dataset": "d1", "metric": "MCC", "metric_key": "mcc", "value": 0.5}
        for model in models
    ]


def test_excludes_only_measured_budget_failures():
    rows = _rows("alpha", "beta", "gamma", "delta")
    policy = {
        "enabled": True,
        "max_runtime_seconds": 100,
        "missing_measurement_policy": "keep_and_flag",
        "models": {
            "alpha": {"runtime_seconds": 101, "measurement_source": "sacct"},
            "beta": {"runtime_seconds": 99, "measurement_source": "sacct"},
        },
    }
    filtered, audit = apply_resource_gate(rows, policy)
    assert {row["model"] for row in filtered} == {"beta", "gamma", "delta"}
    assert audit["excluded_models"] == ["alpha"]
    assert set(audit["flagged_models"]) == {"gamma", "delta"}


def test_gate_is_model_name_agnostic():
    rows = _rows("C_AMPs-predict", "HMD-AMP", "AMPsorter", "leader")
    policy = {
        "enabled": True,
        "max_runtime_seconds": 60,
        "missing_measurement_policy": "keep_and_flag",
        "models": {
            "C_AMPs-predict": {"runtime_seconds": 10, "measurement_source": "sacct job 1"},
            "HMD-AMP": {"runtime_seconds": 11, "measurement_source": "sacct job 2"},
            "AMPsorter": {"runtime_seconds": 12, "measurement_source": "sacct job 3"},
            "leader": {"runtime_seconds": 13, "measurement_source": "sacct job 4"},
        },
    }
    filtered, audit = apply_resource_gate(rows, policy)
    assert len(filtered) == 4
    assert audit["excluded_models"] == []
    assert audit["flagged_models"] == []


def test_unattributed_values_cannot_exclude_a_model():
    rows = _rows("alpha", "beta", "gamma")
    policy = {
        "enabled": True,
        "max_runtime_seconds": 10,
        "missing_measurement_policy": "keep_and_flag",
        "models": {"alpha": {"runtime_seconds": 9999}},
    }
    filtered, audit = apply_resource_gate(rows, policy)
    assert len(filtered) == 3
    alpha = next(check for check in audit["checks"] if check["model"] == "alpha")
    assert alpha["status"] == "eligible_missing_resource_measurement"
    assert "lack measurement_source" in alpha["reasons"][0]
