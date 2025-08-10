# Model Card — NavLNN

- **Type**: Liquid Time-constant network (16–32 hidden)
- **Task**: Safety caps, scan triggers, ramp gains
- **Inputs/Outputs**: see contracts/policy_io.md
- **Training**: Imitation → SafeDAgger; RLHF excluded (safety-only)
- **On-device**: TFLite, NNAPI/GPU optional
- **Telemetry**: log in shadow mode before enable
