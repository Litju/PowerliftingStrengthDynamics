# PSD configuration

Human-authored configuration is **TOML** only, read with the standard library's
`tomllib`. PSD does not introduce YAML or Hydra.

Subdirectories are created when the corresponding capability is implemented:

```text
configs/
├── schema/     # canonical schema build/serialization settings
├── sim/        # PSD-Sim regime configuration
├── benchmark/  # episode, split, and evaluation configuration
└── models/     # baseline model configuration
```

Configuration files are versioned alongside the code that consumes them, and tests
assert that declared schema/tool versions do not drift from the installed package.
