# Releasing AORTA

AORTA is distributed to customers as a versioned, `pip install`-able artifact
attached to a [GitHub Release](https://github.com/ROCm/aorta/releases). AORTA is
a pure-Python package, so a single `py3-none-any` wheel installs on every
platform; PyTorch is intentionally **not** bundled (customers install it from
the ROCm index — see below).

## Maintainer release flow

Releases are automated by [`.github/workflows/release.yml`](../.github/workflows/release.yml),
which triggers on any pushed `v*` tag. To cut a release:

1. **Bump the version.** Edit `version` in `pyproject.toml` (e.g. `0.2.0` ->
   `0.2.1`). Follow [semantic versioning](https://semver.org/).
2. **Commit and merge** the bump to `main` through the normal PR flow.
3. **Tag and push** from the merged commit on `main`:

   ```bash
   git checkout main && git pull
   git tag v0.2.1
   git push origin v0.2.1
   ```

The workflow then:

- builds the wheel + sdist with `python -m build`,
- fails fast if the tag (minus the leading `v`) does not match the
  `pyproject.toml` version, so a release can never disagree with the package
  metadata,
- creates the GitHub Release and uploads `dist/*` (wheel + sdist) with
  auto-generated release notes.

After the workflow finishes, confirm the release page shows
`aorta-<version>-py3-none-any.whl` and run the customer install command below in
a clean virtualenv as a smoke test.

> The tag and the `pyproject.toml` version must match. If they don't, the
> workflow stops before creating the release; bump the version (or fix the tag)
> and push again.

## Customer install flow

PyTorch is installed separately from the ROCm index (it is not part of the
wheel), so customers install in two steps:

```bash
# 1. PyTorch for the target ROCm (adjust the index URL to your ROCm version)
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/rocm7.1/

# 2. AORTA from the release (pin to the version you want)
pip install "aorta @ https://github.com/ROCm/aorta/releases/download/v0.2.0/aorta-0.2.0-py3-none-any.whl"
```

To install with optional extras (for example the hardware-queue tools):

```bash
pip install "aorta[hw-queue] @ https://github.com/ROCm/aorta/releases/download/v0.2.0/aorta-0.2.0-py3-none-any.whl"
```

Replace the version in both the tag path and the filename when installing a
newer release.

## Out of scope (possible follow-ups)

- Publishing to PyPI or a private index — can be layered onto the same workflow
  later without changing the customer-facing package name.
- Signing / attestation of release artifacts.
