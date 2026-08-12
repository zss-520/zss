# Task: propose a bounded self-heal plan

Current registry:

{{var:registry_json}}

Repository documentation and file inventory:

{{var:repository_docs}}

Requirements text:

{{var:requirements_text}}

Environment output:

{{var:environment_output}}

Smoke-test log:

{{var:smoke_log}}

Return one JSON object containing:

```text
diagnosis, pip_install, conda_install, env_setup_commands, registry_updates,
remove_requirement_patterns, retry_smoke
```

Use `registry_updates` only for `python_version`, `dependencies` and `inference_cmd_template`. Do not propose destructive or privileged commands. The runtime performs the final allow-list, dependency and placeholder checks.
