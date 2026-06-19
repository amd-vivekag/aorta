# Releasing AORTA

AORTA is distributed to customers as a versioned, `pip install`-able artifact
attached to a [GitHub Release](https://github.com/ROCm/aorta/releases). AORTA is
a pure-Python package, so a single `py3-none-any` wheel installs on every
platform; PyTorch is intentionally **not** bundled (customers install it from
the ROCm index — see below).

The release version is always read from `version` in `pyproject.toml` — it is
never hard-coded in the workflow — so cutting a new release is just a version
bump plus a trigger.

## Maintainer release flow

Releases are automated by [`.github/workflows/release.yml`](../.github/workflows/release.yml).
Each run builds the artifacts for the current `pyproject.toml` version and
publishes them as a new GitHub Release marked **Latest**.

1. **Bump the version.** Edit `version` in `pyproject.toml` (following
   [semantic versioning](https://semver.org/)) and merge it to `main` through
   the normal PR flow. A new release requires a new version: the workflow
   refuses to re-release a version whose tag already exists.
2. **Trigger the release**, either way:
   - **Push a version tag** matching the new version:

     ```bash
     git checkout main && git pull
     git tag vX.Y.Z      # X.Y.Z must equal the pyproject.toml version
     git push origin vX.Y.Z
     ```

   - **Run it manually** from the GitHub UI (*Actions -> Release -> Run
     workflow*). The workflow reads the version from `pyproject.toml`, then
     creates and pushes the matching `vX.Y.Z` tag for you.

The workflow then:

- reads the version from `pyproject.toml`,
- **before building**, fails fast if a pushed tag does not match that version
  (so a release can never disagree with the package metadata),
- builds the wheel + sdist with `python -m build`,
- creates the GitHub Release named `AORTA X.Y.Z`, marks it **Latest**, and
  uploads the wheel + sdist as release assets with auto-generated notes.

After the workflow finishes, confirm the [latest release](https://github.com/ROCm/aorta/releases/latest)
shows `aorta-X.Y.Z-py3-none-any.whl` plus the sdist, and run the customer
install command below in a clean virtualenv as a smoke test.

## Customer install flow

PyTorch is installed separately from the ROCm index (it is not part of the
wheel), so customers install in two steps. Replace `X.Y.Z` with the release
version you want — browse [the releases page](https://github.com/ROCm/aorta/releases)
(the newest is tagged **Latest**) to find it:

```bash
# 1. PyTorch for the target ROCm (adjust the index URL to your ROCm version)
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/rocm7.1/

# 2. AORTA from the release (pin to the version you want)
pip install "aorta @ https://github.com/ROCm/aorta/releases/download/vX.Y.Z/aorta-X.Y.Z-py3-none-any.whl"
```

To install with optional extras (for example the hardware-queue tools):

```bash
pip install "aorta[hw-queue] @ https://github.com/ROCm/aorta/releases/download/vX.Y.Z/aorta-X.Y.Z-py3-none-any.whl"
```

## Out of scope (possible follow-ups)

- Publishing to PyPI or a private index — can be layered onto the same workflow
  later without changing the customer-facing package name.
- Signing / attestation of release artifacts.
