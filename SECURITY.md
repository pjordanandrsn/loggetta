# Security policy

## Reporting

**Do not open a public issue.** Report by email:

- `security@cerinamroth.com` — encrypt to the
  [published key](https://cerinamroth.com/.well-known/cerinamroth-pubkey.asc)
  (`gpg --locate-keys security@cerinamroth.com`)

Canonical contact: [`security.txt`](https://cerinamroth.com/.well-known/security.txt).
The [disclosure policy](https://cerinamroth.com/policy/) applies: good-faith
research, coordinated disclosure, no publication before a reasonable remediation
window.

## In scope

Loggetta reads model configs, tokenizers and your datasets, plans placement, runs
a backend, and writes a plan and a run report. The surface is what it parses,
what it executes, and what its reports claim.

- **Dataset and config parsing.** A crafted dataset file or model config causing
  arbitrary code execution, or reads and writes outside the paths you gave.
- **Remote code.** Loggetta passes `trust_remote_code` only when you set
  `--trust-remote-code`. Any path that executes a model repository's code without
  that flag is in scope.
- **Report integrity.** A run report that records a check as passed, or a
  measurement, that did not happen.

## Not in scope

- An estimate that misses. Plans are estimates, not an out-of-memory guarantee;
  report a miss as an issue.
- Code you opted into with `--trust-remote-code`.
- Vulnerabilities in the runtime or kernels (report to
  [experts4bit-qlora](https://github.com/pjordanandrsn/experts4bit-qlora/security/policy)
  or [grouped-nf4-gemm](https://github.com/pjordanandrsn/grouped-nf4-gemm/security/policy)),
  or in torch, transformers or NVIDIA drivers (report upstream).

## What to expect

Acknowledgement when seen — one maintainer, best-effort, not an SLA. Then
confirmation or a reasoned disagreement, a fix, a release, and credit unless you
prefer otherwise.
