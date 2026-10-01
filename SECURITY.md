# Security Policy

## Reporting a vulnerability

**Do not open a public issue for a security vulnerability.**

PSD has GitHub **private vulnerability reporting** enabled. Use
[Report a vulnerability](https://github.com/Litju/PowerliftingStrengthDynamics/security/advisories/new)
on the repository's Security tab to submit a private report. That channel is private
between you and the maintainer until an advisory is published.

Please include:

* affected version or commit (tag or SHA);
* the platform (Windows or Linux) and Python version;
* a minimal reproduction, ideally the smallest input or command that shows the issue;
* the observed impact, and your assessment of severity;
* any suggested remediation.

If you cannot use GitHub private vulnerability reporting, contact the maintainer through
the GitHub account that owns the repository
([github.com/Litju](https://github.com/Litju)) and ask for a private channel.

You should receive an acknowledgement within a few days. Please allow time for a fix and
a release before public disclosure. We will credit reporters in the advisory unless you
prefer otherwise.

### What is in scope

* Code in this repository that executes untrusted input, writes outside its declared
  path boundary, or mishandles provenance, privacy, or integrity of artifacts.
* Path handling that could escape `PSD_DATA_ROOT` or the repository.
* Deserialization of untrusted files by the library or CLI.
* Exposure of secrets, credentials, or private keys through tracked files, logs,
  manifests, or build artifacts.
* Any defect that would corrupt canonical artifacts or silently drop provenance,
  missingness, or source semantics.

### What is out of scope

* Vulnerabilities in third-party dependencies with no demonstrated impact through PSD
  code — report those upstream, though let us know so we can pin or patch.
* Findings that require an attacker who already has your user account, your machine, or
  write access to the repository.
* Scientific or modeling correctness disputes. Those belong in public issues and follow
  [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
* Data-quality or provenance-annotation mistakes that are not security issues.
* Denial of service from deliberately adversarial multi-terabyte inputs; PSD is a research
  tool and does not promise resource isolation.

## Supported versions

PSD is a **pre-alpha research project**. There is no stable release series.

| Version | Supported |
| --- | --- |
| `main` | Yes — actively developed; fixes land here first |
| Latest tagged pre-alpha release | Yes, best effort |
| Anything older than the latest tag | No |
| Unreleased/untagged local builds | No |

There is no long-term-support branch during Alpha. Breaking changes to the canonical
schema are expected: the schema carries a `schema_version` (currently
`psd-canonical/0.1.0`) precisely so artifacts are never silently reinterpreted. Pin
commits and record schema versions in manifests.

## Data and privacy

PSD is designed to hold personal data about athletes. Please report, and treat as
security-relevant:

* accidental inclusion of identifiable athlete data in the repository, logs, or
  published artifacts;
* exposure of free-text athlete notes;
* failure to preserve consent, provenance, or redistribution constraints recorded in
  dataset manifests;
* any path that could write athlete data outside the declared external data root.

## Out-of-scope disclosures and researcher behavior

We ask that researchers follow good-faith practices:

* work against locally generated or synthetic data, not real athlete exports, unless the
  dataset's own consent and provenance permit it;
* avoid exfiltrating data;
* avoid persistent changes to a contributor's repository or machine;
* give the maintainer a reasonable window to ship a fix before disclosure.
