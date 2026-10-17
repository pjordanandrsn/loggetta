### Tests: a default config that fails transformers' own validation is skipped by name

The refusal test for unsupported default configs built each type with `AutoConfig.for_model`. transformers 5.20.1's
default `gemma4_assistant` config fails its own validator ("hidden_size_per_layer_input must be 0, got 256"), so the test
failed before any loggetta code ran. Building the default config is now guarded: such a failure skips with transformers'
version and the validator's message, as a type missing from `CONFIG_MAPPING` already did. What loggetta does with a
config that builds is still asserted. Tests only; no runtime change.
