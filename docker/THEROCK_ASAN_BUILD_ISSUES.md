# TheRock HOST_ASAN + PyTorch Docker Build Issues & Resolutions

This document tracks all build issues encountered while creating
`Dockerfile.therock-host-asan-pytorch` — a Docker image that builds ROCm/HIP
from source via [TheRock](https://github.com/ROCm/TheRock) with host-only
AddressSanitizer (HOST_ASAN), then builds PyTorch against that instrumented stack.

Reference Dockerfile: `docker/Dockerfile.rocm70_2-ubuntu-pytorch` (working
PyTorch + ROCm build using pre-built debian packages).

---

## Issue 1: Missing ROCm path environment variables

**Symptom:** PyTorch's CMake could not find ROCm components (rocblas, hipblas,
miopen, etc.) during configuration, even though `ROCM_HOME=/opt/rocm` was set.

**Root cause:** The working Dockerfile installs ROCm via debian packages that
ship CMake config files and pkg-config entries. TheRock's `cmake --install`
produces a different layout. PyTorch's `find_package()` calls need
`CMAKE_PREFIX_PATH` to locate components, and `HIP_PATH` / `ROCM_PATH` are
also consulted by various FindHIP/FindROCm scripts.

**Resolution:** Added three additional environment variables alongside `ROCM_HOME`:

```dockerfile
ENV ROCM_PATH=${ROCM_INSTALL_PREFIX}
ENV HIP_PATH=${ROCM_INSTALL_PREFIX}
ENV CMAKE_PREFIX_PATH=${ROCM_INSTALL_PREFIX}
```

---

## Issue 2: pip cannot uninstall debian-managed pip on Ubuntu 24.04

**Symptom:**
```
ERROR: Cannot uninstall pip 24.0, RECORD file not found.
Hint: The package was installed by debian.
```

**Root cause:** Ubuntu 24.04 enforces PEP 668 — system-installed pip has no
RECORD file, so `pip install --upgrade pip` fails when trying to uninstall
the existing version. Using `--break-system-packages` doesn't help with this
specific error.

**Resolution:** Use a Python venv for PyTorch build dependencies. The venv
has its own pip that can be freely upgraded:

```dockerfile
RUN python3 -m venv /opt/pytorch-venv && \
    /opt/pytorch-venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/pytorch-venv/bin/pip install --no-cache-dir pyyaml typing_extensions numpy pandas

ENV PATH=/opt/pytorch-venv/bin:${PATH}
```

---

## Issue 3: AOTriton configure fails — missing liblzma

**Symptom:**
```
CMake Error at /usr/share/cmake-4.2/Modules/FindPkgConfig.cmake:1203 (message):
    None of the required 'liblzma' found
FAILED: aotriton/src/aotriton_runtime-stamp/aotriton_runtime-configure
```

**Root cause:** PyTorch v2.11 configures the `aotriton_runtime` ExternalProject
even when `DISABLE_AOTRITON=1` is set (the flag only disables runtime usage,
not the build configure step). AOTriton's CMakeLists.txt requires `liblzma`
via `pkg_search_module(liblzma)`.

**Resolution:** Added `liblzma-dev` to the apt package list:

```dockerfile
RUN apt-get update && apt-get install -y \
    ...
    liblzma-dev \
    ...
```

---

## Issue 4: TheRock rocRoller link failure — ASAN symbol mismatch

**Symptom:**
```
[rocRoller] ld.lld: error: undefined symbol: __asan_option_detect_stack_use_after_return
>>> referenced by ... in archive /build/TheRock/build/third-party/yaml-cpp/dist/lib/libyaml-cpp.a
[rocRoller] ld.lld: error: undefined symbol: __asan_stack_malloc_4
[rocRoller] ld.lld: error: undefined symbol: __asan_set_shadow_f5
```

**Root cause:** The `linux-release-host-asan` preset sets
`THEROCK_SANITIZER=HOST_ASAN` globally, which adds `-fsanitize=address` to
the global `CMAKE_C_FLAGS` / `CMAKE_CXX_FLAGS`. This affects ALL compiled
code — including third-party dependencies like `yaml-cpp`.

We attempted to selectively disable ASAN on individual components (rocRoller,
composable_kernel, etc.) via `-D<component>_SANITIZER=OFF`. However,
per-component `_SANITIZER=OFF` only removes ASAN from TheRock's own managed
sub-projects' build flags. Third-party static libraries like `libyaml-cpp.a`
always inherit the global ASAN flags and are compiled with ASAN symbols.

When rocRoller (SANITIZER=OFF, no `-fsanitize=address` in link flags) tried to
link against the ASAN-instrumented `libyaml-cpp.a`, the ASAN runtime symbols
were unresolvable — the linker doesn't pull in `libclang_rt.asan` without
`-fsanitize=address`.

**Resolution:** Use the `linux-release-host-asan` preset as designed — let all
components get HOST_ASAN consistently. Do NOT try to selectively disable ASAN
on individual components. Only keep the minimal overrides that the upstream
`CMakePresets.json` already defines (amd-llvm, hipcc, hipify, etc. are
excluded from ASAN in the preset itself) plus `therock-SuiteSparse_SANITIZER=OFF`
and `amdsmi_SANITIZER=OFF` which were needed to avoid other build issues:

```dockerfile
cmake --preset linux-release-host-asan \
    -B build \
    -DTHEROCK_AMDGPU_FAMILIES=${THEROCK_AMDGPU_FAMILIES} \
    -DTHEROCK_ENABLE_DC_TOOLS=OFF \
    -DBUILD_TESTING=OFF \
    -Dtherock-SuiteSparse_SANITIZER=OFF \
    -Damdsmi_SANITIZER=OFF \
    ${MINIMAL_FLAGS}
```

**Key learning:** You cannot apply ASAN to only the HIP runtime when using
TheRock's build system. The global ASAN flags propagate to all code including
third-party deps, and any mismatch between ASAN-instrumented static libraries
and non-ASAN executables causes linker errors.

---

## Issue 5: PyTorch build fails — `nccl_device.h` not found

**Symptom:**
```
/build/pytorch/torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp:14:10:
    fatal error: 'nccl_device.h' file not found
   14 | #include <nccl_device.h>
```

Build was at `[12174/12781]` — very near the end of the PyTorch compilation.

**Root cause:** PyTorch v2.11 requires `nccl_device.h` for the NCCL Symmetric
Memory feature (`NCCLSymmetricMemory.cu`). When ROCm is installed via debian
packages (as in the working Dockerfile), RCCL ships with proper CMake config
files that PyTorch auto-detects via `find_package()`. With TheRock's
`cmake --install`, the install layout and CMake config files differ — PyTorch's
NCCL detection falls back to building its own bundled NCCL submodule, which
doesn't provide `nccl_device.h`.

**Resolution (partial):** Initially added `USE_SYSTEM_NCCL=1` with RCCL paths:

```dockerfile
ENV USE_SYSTEM_NCCL=1
ENV NCCL_ROOT=/opt/rocm
ENV NCCL_INCLUDE_DIR=/opt/rocm/include
ENV NCCL_LIB_DIR=/opt/rocm/lib
```

These are placed after `git submodule update --init --recursive` to avoid
invalidating Docker layer cache for the clone/submodule steps.

However, this alone did not fix the issue — see Issue 8.

---

## Issue 6: PyTorch cloned into TheRock directory

**Symptom:** PyTorch source ended up at `/build/TheRock/pytorch/` instead of
`/build/pytorch/`, causing confusing paths in build logs.

**Root cause:** No `WORKDIR` directive between the TheRock build (which sets
`WORKDIR /build/TheRock`) and the PyTorch `git clone`.

**Resolution:** Added `WORKDIR /build` before the PyTorch clone:

```dockerfile
WORKDIR /build

RUN git clone --branch ${PYTORCH_GIT_REF} --single-branch \
    https://github.com/pytorch/pytorch.git
```

---

## Issue 7: Docker build log truncated at 2 MiB

**Symptom:**
```
[output clipped, log limit 2MiB reached]
```

The actual compilation error was hidden in the middle of the output, making
debugging impossible. The CK SDPA kernel compilation generates thousands of
repetitive warning lines per `.hip.o` file, quickly exhausting the 2 MiB
BuildKit log buffer.

**Root cause:** Docker BuildKit has a default per-step log buffer limit of
2 MiB. The verbose CK SDPA kernel warnings fill this buffer before the build
completes.

**Workaround:** Use `--progress=plain` to stream full output to the terminal
(bypasses the internal buffer), and pipe to `tee` to capture everything:

```bash
docker compose -f docker/docker-compose.build.yaml build --progress=plain 2>&1 \
    | tee stdout.asan_docker_build.log
```

The `tee` file captures the full terminal stream regardless of BuildKit's
internal buffer limit. The `[output clipped]` message only affects what
BuildKit stores internally — `--progress=plain` streams everything to stdout
as it happens.

To also increase the internal buffer (useful for `docker buildx`):

```bash
docker buildx create \
    --name biglog \
    --driver docker-container \
    --driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=104857600 \
    --use
```

---

## Issue 8: `nccl_device.h` still not found despite `USE_SYSTEM_NCCL=1`

**Symptom:** Same error as Issue 5 — still `fatal error: 'nccl_device.h' file not found`
even after setting `USE_SYSTEM_NCCL=1` and `NCCL_INCLUDE_DIR=/opt/rocm/include`.
Build failed at `[12175/12781]` after ~75 minutes of compilation.

**Root cause:** `USE_SYSTEM_NCCL=1` told PyTorch to search `/opt/rocm/include`
for NCCL headers (visible in the `-I/opt/rocm/include` compiler flags), but
**`nccl_device.h` simply doesn't exist there**. TheRock's RCCL
`cmake --install` installs `nccl.h` and `rccl.h` but omits `nccl_device.h` —
a device-level header that lives in RCCL's source tree
(`src/include/nccl_device.h`) and is not part of the standard install targets.

The ROCm debian packages (used in `Dockerfile.rocm70_2-ubuntu-pytorch`)
include `nccl_device.h` as part of the `rccl-dev` package, which is why the
working Dockerfile doesn't need any special handling.

**Resolution:** After TheRock's `cmake --install`, copy `nccl_device.h` from
the RCCL source tree in TheRock's build directory to the install prefix:

```dockerfile
RUN find /build/TheRock -name "nccl_device.h" -type f | head -1 | \
    xargs -I{} cp {} ${ROCM_INSTALL_PREFIX}/include/ && \
    ls -la ${ROCM_INSTALL_PREFIX}/include/nccl_device.h
```

This step is placed immediately after `cmake --install` so it fails fast
(in seconds) if the header can't be found, rather than after a 75-minute
PyTorch build.

**Why not just `USE_SYSTEM_NCCL`?** The env var only controls which NCCL
library PyTorch links against (system vs bundled submodule). It doesn't
help if the header file is physically missing from the include path.
Both the system RCCL and the bundled NCCL need the header to be present;
the difference is that TheRock's install step doesn't place it there.

---

## Issue 9: `nccl_device/impl/comm__funcs.h` not found (incomplete RCCL header copy)

**Symptom:**
```
/opt/rocm/include/nccl_device.h:7:10: fatal error: 'nccl_device/impl/comm__funcs.h' file not found
    7 | #include "nccl_device/impl/comm__funcs.h"
```

Build was at `[12173/12781]` — same file (`NCCLSymmetricMemory.cu`) as Issue 8.

**Root cause:** The Issue 8 fix only copied the single `nccl_device.h` file
from RCCL's source tree. However, `nccl_device.h` uses `#include` directives
that reference a `nccl_device/` subdirectory containing implementation headers
(`nccl_device/impl/comm__funcs.h`, etc.). Without the entire subdirectory tree,
the compiler finds `nccl_device.h` but cannot resolve its transitive includes.

**Resolution:** Instead of copying just the single file, find the RCCL include
directory and copy both `nccl_device.h` and the entire `nccl_device/`
subdirectory:

```dockerfile
RUN RCCL_SRC=$(find /build/TheRock -path "*/rccl/src/include/nccl_device.h" -type f | head -1 | xargs dirname) && \
    cp ${RCCL_SRC}/nccl_device.h ${ROCM_INSTALL_PREFIX}/include/ && \
    cp -r ${RCCL_SRC}/nccl_device ${ROCM_INSTALL_PREFIX}/include/ && \
    ls -la ${ROCM_INSTALL_PREFIX}/include/nccl_device.h && \
    ls ${ROCM_INSTALL_PREFIX}/include/nccl_device/
```

**Key learning:** When copying headers from a source tree, always check whether
the header has transitive includes pointing to sibling directories. A single
`cp` of one file is rarely sufficient for C/C++ header hierarchies.

---

## Issue 10: `../comm_tmp.h` not found (incomplete RCCL header directory copy)

**Symptom:**
```
/opt/rocm/include/nccl_device/impl/comm__types.h:9:10: fatal error: '../comm_tmp.h' file not found
    9 | #include "../comm_tmp.h"
```

Two files failed: `nccl_extension.cu` and `NCCLSymmetricMemory.cu` at
`[11894/12781]` and `[11895/12781]`.

**Root cause:** The Issue 9 fix copied `nccl_device.h` and the `nccl_device/`
subdirectory, but RCCL's device headers use relative `#include` paths that
reference sibling files at various directory levels. Specifically,
`nccl_device/impl/comm__types.h` includes `../comm_tmp.h`, which resolves to
`nccl_device/comm_tmp.h` — a file that exists in RCCL's `src/include/`
directory but was either not part of the `nccl_device/` subdirectory or was
generated during the build.

Cherry-picking individual files or subdirectories from RCCL's source include
tree doesn't work because the headers have deep cross-references with relative
paths.

**Resolution:** Copy the **entire contents** of RCCL's `src/include/` directory
to the install prefix, preserving the full relative include structure:

```dockerfile
RUN RCCL_INC=$(find /build/TheRock -path "*/rccl/src/include/nccl_device.h" -type f | head -1 | xargs dirname) && \
    cp -r ${RCCL_INC}/* ${ROCM_INSTALL_PREFIX}/include/ && \
    ls -la ${ROCM_INSTALL_PREFIX}/include/nccl_device.h && \
    ls ${ROCM_INSTALL_PREFIX}/include/nccl_device/
```

**Key learning:** When a source tree has headers with relative `#include`
paths (`../`, `../../`, etc.), you must copy the entire include subtree — not
individual files or directories. The relative paths encode assumptions about
the directory layout.

---

## Issue 11: `comm_tmp.h` still not found — generated file not in source tree

**Symptom:** Same as Issue 10:
```
/opt/rocm/include/nccl_device/impl/comm__types.h:9:10: fatal error: '../comm_tmp.h' file not found
```

**Root cause:** The Issue 10 fix (`cp -r ${RCCL_INC}/*`) copies the entire
RCCL `src/include/` directory, but `comm_tmp.h` is a **generated** header —
it's created during RCCL's CMake build process (likely by a code generator
like `generate.py`) and placed in the **build** directory, not the source
tree. Since it never existed in `rccl/src/include/nccl_device/`, the
wildcard copy didn't include it.

**Resolution:** After copying the source headers, search the TheRock build
tree for build-generated RCCL device headers (like `comm_tmp.h`) and merge
them into the install prefix:

```dockerfile
RUN RCCL_INC=$(find /build/TheRock -path "*/rccl/src/include/nccl_device.h" \
        -type f | head -1 | xargs dirname) && \
    cp -r ${RCCL_INC}/* ${ROCM_INSTALL_PREFIX}/include/ && \
    for f in $(find /build/TheRock -path "*/rccl*" -name "comm_tmp.h" -type f); do \
        DEST_DIR=${ROCM_INSTALL_PREFIX}/include/$(dirname "${f}" | grep -oP 'nccl_device.*'); \
        mkdir -p "${DEST_DIR}" && cp -n "${f}" "${DEST_DIR}/"; \
    done && \
    test -f ${ROCM_INSTALL_PREFIX}/include/nccl_device/comm_tmp.h || \
        (echo "ERROR: comm_tmp.h not found" && exit 1)
```

The step now includes a validation check: if `comm_tmp.h` can't be found
anywhere in the TheRock build tree, the build fails immediately with a
diagnostic listing all `comm_tmp.h` locations, instead of failing 75 minutes
later during PyTorch compilation.

**Key learning:** Header directories for projects like RCCL contain a mix of
source-tree headers and build-generated headers. Copying only the source
include directory misses generated files. You must merge both source and
build directories.

---

## Issue 12: `core_tmp.h` not found — more generated headers missing

**Symptom:**
```
/opt/rocm/include/nccl_device/impl/../comm_tmp.h:9:10: fatal error: 'core_tmp.h' file not found
```

**Root cause:** Issues 8–11 attempted to copy RCCL device headers piecemeal:
first `nccl_device.h`, then `nccl_device/`, then `comm_tmp.h`. Each fix
uncovered another missing generated header. The RCCL build's hipify step
(`hipify-perl`) and code generator (`generate.py`) produce multiple `*_tmp.h`
files (`comm_tmp.h`, `core_tmp.h`, `ptr_tmp.h`, etc.) in the **hipified build
tree** at `${CMAKE_BINARY_DIR}/hipify/src/include/nccl_device/`. Copying from
the source tree or searching for individual files by name is a losing game.

**Resolution:** Copy from the **hipified build directory** instead of the
source tree. The path `*/hipify/src/include/` in TheRock's build tree contains
the complete set of RCCL device headers — both hipified source headers and
all generated `*_tmp.h` files:

```dockerfile
RUN HIPIFY_INC=$(find /build/TheRock -path "*/hipify/src/include/nccl_device.h" \
        -type f | head -1 | xargs dirname) && \
    cp -r ${HIPIFY_INC}/nccl_device.h ${ROCM_INSTALL_PREFIX}/include/ && \
    cp -r ${HIPIFY_INC}/nccl_device ${ROCM_INSTALL_PREFIX}/include/ && \
    find ${ROCM_INSTALL_PREFIX}/include/nccl_device -type f | sort
```

**Key learning:** RCCL's build produces a parallel "hipified" include tree
under `<build>/hipify/src/include/` that mirrors `src/include/` but with
generated files added. Always copy from this directory — not from the source
tree — to get the complete header set. This is exactly what RCCL's
`CMakeLists.txt` should do in its `install()` section (see analysis above).

---

## Issue 13: `nccl.h` not found — RCCL build-internal header not installed

**Symptom:**
```
/opt/rocm/include/nccl_device/impl/../core_tmp.h:9:10: fatal error: 'nccl.h' file not found
```

**Root cause:** RCCL's `CMakeLists.txt` generates two copies of the main
header from `src/nccl.h.in`:

```cmake
configure_file(src/nccl.h.in ${PROJECT_BINARY_DIR}/include/rccl/rccl.h)  # installed
configure_file(src/nccl.h.in ${PROJECT_BINARY_DIR}/include/nccl.h)       # internal only
```

Only `rccl.h` is installed (to `<prefix>/include/rccl/rccl.h`). The internal
`nccl.h` is used during RCCL's build via `-I${PROJECT_BINARY_DIR}/include`
but never installed. The device headers (e.g. `core_tmp.h`) `#include "nccl.h"`
by that name, which fails when they're placed in `/opt/rocm/include/` without
the internal `nccl.h`.

The ROCm debian packages ship both `rccl.h` and `nccl.h` in their `-dev`
package, so this problem doesn't appear with package-based installs.

**Resolution:** Find the generated `nccl.h` in the RCCL build directory and
copy it to the install prefix:

```dockerfile
RCCL_NCCL_H=$(find /build/TheRock -path "*/rccl*/include/nccl.h" \
    ! -path "*/src/*" -type f | head -1) && \
cp ${RCCL_NCCL_H} ${ROCM_INSTALL_PREFIX}/include/nccl.h
```

**Key learning:** RCCL has a split-header design: `rccl.h` (public, installed)
and `nccl.h` (internal, identical content, not installed). When exposing
internal device headers for consumers like PyTorch, the internal `nccl.h`
must also be installed.

---

## Issue 14: `core.h` not found — nccl_device headers reference parent includes

**Symptom:**
```
/opt/rocm/include/nccl_device/impl/../ptr.h:9:10: fatal error: 'core.h' file not found
```

**Root cause:** The file listing from the copy step revealed that `core.h`
and `comm.h` are **not** in the `hipify/src/include/nccl_device/` directory.
The generate.py script produces `core_tmp.h` and `comm_tmp.h` as replacements,
but the original `core.h` lives at `hipify/src/include/core.h` (one level up).

During RCCL's build, CMake adds both directories to the include path:
```cmake
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include/nccl_device)
```

So `#include "core.h"` in `nccl_device/ptr.h` resolves via the `-I` path to
`src/include/core.h`. When we only copied `nccl_device/` to `/opt/rocm/include/`,
the parent-level headers (`core.h`, `comm.h`, `device.h`, etc.) were missing.

**Resolution:** Copy the **entire** `hipify/src/include/` tree to the install
prefix (using `-n` to avoid overwriting existing files from TheRock's install):

```dockerfile
RUN HIPIFY_INC=$(find /build/TheRock \
        -path "*/hipify/src/include/nccl_device.h" \
        -type f | head -1 | xargs dirname) && \
    cp -rn ${HIPIFY_INC}/* ${ROCM_INSTALL_PREFIX}/include/
```

This adds RCCL's internal headers (core.h, comm.h, device.h, etc.) alongside
the device headers. The `-n` flag ensures already-installed headers from
TheRock's `cmake --install` are not overwritten.

---

## Issue 15: Pre-flight header check to catch all missing includes at once

**Symptom:** Each missing RCCL header was only discovered after a 75-minute
PyTorch build reached the `NCCLSymmetricMemory.cu` file at step `[12173/12781]`.
Issues 8–14 each required a full rebuild cycle.

**Root cause:** No validation step between installing RCCL headers and starting
the PyTorch build.

**Resolution:** Added two fixes:

1. **Symlinks for ambiguous includes**: RCCL's `nccl_device/ptr.h` includes
`"core.h"` which exists at the parent level (`src/include/core.h`) but not
in `nccl_device/`. During RCCL's build this works via `-I` paths, but PyTorch
doesn't have that path. Fixed by symlinking:

```dockerfile
for h in core.h comm.h device.h; do
    if [ -f ${ROCM_INSTALL_PREFIX}/include/${h} ] && \
       [ ! -f ${ROCM_INSTALL_PREFIX}/include/nccl_device/${h} ]; then
        ln -s ../${h} ${ROCM_INSTALL_PREFIX}/include/nccl_device/${h}
    fi
done
```

2. **Pre-flight compilation check**: A 2-second step that test-compiles
`#include "nccl_device.h"` with the same include paths PyTorch will use.
Any missing transitive include fails immediately:

```dockerfile
RUN echo '#include "nccl_device.h"' > /tmp/test_nccl_device.cpp && \
    ${ROCM_INSTALL_PREFIX}/llvm/bin/clang++ -fsyntax-only -x c++ \
        -I${ROCM_INSTALL_PREFIX}/include \
        -I${ROCM_INSTALL_PREFIX}/include/nccl_device \
        -ferror-limit=0 \
        /tmp/test_nccl_device.cpp
```

With `-ferror-limit=0`, the compiler reports ALL missing headers at once
instead of stopping at the first one.

---

## Issue 16: Symlinks pointed to wrong `core.h` — RCCL internal vs device header

**Symptom:** Pre-flight check fails with errors from deep RCCL internals:
```
/opt/rocm/include/core.h:39: In file included: alloc.h
/opt/rocm/include/core.h:45: In file included: nvtx.h → roctx.h → device.h
/opt/rocm/include/device.h:29:10: fatal error: 'nccl_tuner.h' file not found
```

**Root cause:** Issue 15's fix created symlinks:
```
nccl_device/core.h -> ../core.h    (WRONG!)
nccl_device/comm.h -> ../comm.h    (WRONG!)
```

RCCL has **two completely different files** named `core.h`:
- `src/include/nccl_device/core.h` — lightweight device header (~160 lines,
  includes only `coop.h` and `utility.h`)
- `src/include/core.h` — heavy RCCL internal header (~500+ lines, pulls in
  `alloc.h`, `nvtx.h`, `device.h`, `nccl_tuner.h`, and the entire RCCL
  internals)

The symlink pointed `nccl_device/core.h` at the wrong one, causing the
include chain to explode into RCCL's full internal header tree, eventually
failing on `nccl_tuner.h` (which lives in `src/include/plugin/`).

Similarly, Issue 14's approach of copying the entire `hipify/src/include/*`
was wrong — it dumped RCCL's internal headers (`core.h`, `alloc.h`, `nvtx.h`,
`device.h`, etc.) into `/opt/rocm/include/`, polluting the install prefix.

**Resolution:** 
1. Removed the symlinks and the bulk `cp -rn ${HIPIFY_INC}/*` approach
2. Copy only `nccl_device.h` and `nccl_device/` from the hipified build tree
   (which has the generated `*_tmp.h` files)
3. Copy `nccl_device/core.h` and `nccl_device/comm.h` from the **RCCL source
   tree** (not the hipified tree, which doesn't produce them; not the parent
   directory, which has different files with the same name)

```dockerfile
RCCL_SRC_INC=$(find /build/TheRock \
    -path "*/rccl/src/include/nccl_device/core.h" \
    -type f | head -1 | xargs dirname) && \
for h in core.h comm.h; do
    cp ${RCCL_SRC_INC}/${h} ${ROCM_INSTALL_PREFIX}/include/nccl_device/${h}
done
```

**Key learning:** When multiple directories contain files with the same name,
never use symlinks or bulk copies without verifying file identity. RCCL's
`src/include/core.h` and `src/include/nccl_device/core.h` have completely
different contents despite the same filename.

---

## Summary of all fixes applied

| # | Issue | Fix | Dockerfile location |
|---|-------|-----|---------------------|
| 1 | Missing ROCm path vars | Added `ROCM_PATH`, `HIP_PATH`, `CMAKE_PREFIX_PATH` | Stage 3: ROCm environment |
| 2 | pip uninstall failure on Ubuntu 24.04 | Use Python venv instead of system pip | Stage 4: before PyTorch build |
| 3 | AOTriton missing liblzma | Added `liblzma-dev` to apt packages | Stage 1: System packages |
| 4 | ASAN symbol mismatch in TheRock | Use preset as designed; don't selectively disable ASAN | Stage 2: TheRock configure |
| 5 | `nccl_device.h` not found (initial) | Set `USE_SYSTEM_NCCL=1` + RCCL paths (partial fix) | Stage 4: before `setup.py develop` |
| 6 | PyTorch cloned in wrong directory | Added `WORKDIR /build` before clone | Stage 4: before git clone |
| 7 | Docker log truncated at 2 MiB | Use `--progress=plain` with `tee` | Build command (not Dockerfile) |
| 8 | `nccl_device.h` still missing | Copy from RCCL source tree after TheRock install (partial) | Stage 2: after `cmake --install` |
| 9 | `nccl_device/` subdirectory missing | Copy entire `nccl_device/` dir alongside `nccl_device.h` (partial) | Stage 2: after `cmake --install` |
| 10 | `../comm_tmp.h` relative include missing | Copy entire RCCL `src/include/*` contents (partial) | Stage 4: after submodule init |
| 11 | `comm_tmp.h` is build-generated, not in source | Find & merge generated headers from RCCL build dir (partial) | Stage 4: after submodule init |
| 12 | `core_tmp.h` + more generated headers missing | Copy from `hipify/src/include/` build tree | Stage 4: after submodule init |
| 13 | `nccl.h` not installed (internal-only header) | Copy generated `nccl.h` from RCCL build dir | Stage 4: after submodule init |
| 14 | `core.h` missing (parent-level internal header) | Copy entire `hipify/src/include/*` with `cp -rn` | Stage 4: after submodule init |
| 15 | Repeated build cycles for each missing header | Pre-flight `clang++ -fsyntax-only` check | Stage 4: after header copy |
| 16 | Symlinks to wrong `core.h` (internal vs device) | Copy from RCCL source tree, not symlink to parent | Stage 4: after header copy |
| 17 | `railGinBarrierCount` missing in RCCL struct | Patch `nccl_dev_cap.hpp` to raise SYMMEM_DEVICE threshold | Stage 4: after PyTorch clone |
| 18 | `libdrm/drm.h` not found (`rocm_smi/kfd_ioctl.h`) | Added `libdrm-dev` to apt packages | Stage 1: System packages |
| 19 | `import torch` segfaults (ASAN not preloaded) | `LD_PRELOAD` ASAN runtime in verify step | Stage 4: verify installations |

---

## Issue 17: RCCL/NCCL API version mismatch — `railGinBarrierCount`

**Error:**
```
/build/pytorch/torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp:80:10:
  error: no member named 'railGinBarrierCount' in 'ncclDevCommRequirements'
    reqs.railGinBarrierCount = gin_barrier_count;
    ~~~  ^
```

**Root cause:** PyTorch v2.11.0-rc2 file
`torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp` defines:

```cpp
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
#define NCCL_HAS_SYMMEM_DEVICE_SUPPORT
#include <nccl_device.h>
#endif
```

TheRock's RCCL (develop branch) reports **NCCL version 2.28.3**. This triggers
`NCCL_HAS_SYMMEM_DEVICE_SUPPORT`, which compiles `nccl_devcomm_manager.hpp`.
That file uses `reqs.railGinBarrierCount`, but that struct field was only added
in **NCCL 2.28.7**. RCCL 2.28.3 has the `ncclDevCommRequirements` struct but
without `railGinBarrierCount`.

The **working** Dockerfile (`Dockerfile.rocm70_2-ubuntu-pytorch`) uses ROCm
7.0.2 packages, which ship RCCL ~2.22.x. At that version, both
`NCCL_HAS_SYMMEM_SUPPORT` (>= 2.27.0) and `NCCL_HAS_SYMMEM_DEVICE_SUPPORT`
(>= 2.28.0) are disabled, so the entire symmetric-memory device code is skipped.

**Fix:** Patch `nccl_dev_cap.hpp` to raise the `NCCL_HAS_SYMMEM_DEVICE_SUPPORT`
threshold from 2.28.0 to 2.29.0. This prevents the device-communicator code
from compiling while keeping basic symmetric memory support (>= 2.27.0):

```dockerfile
RUN sed -i 's/NCCL_VERSION(2, 28, 0)/NCCL_VERSION(2, 29, 0)/' \
    pytorch/torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp
```

Additionally, the complex RCCL device-header copy (nccl_device.h, nccl_device/,
generated *_tmp.h files, nccl.h from build tree) and pre-flight clang++ check
are no longer needed because `nccl_device.h` is only `#include`d inside the
`NCCL_HAS_SYMMEM_DEVICE_SUPPORT` guard. Replaced the entire header-copy block
with a simple symlink: `ln -s rccl.h nccl.h`.

**Key learning:** When building PyTorch against an RCCL version that tracks
NCCL but lags behind on specific features, the version-gated macros in
PyTorch may enable code that references API additions not yet in RCCL.
Always check the RCCL version's actual API surface against what PyTorch
expects, especially for features marked "available since NCCL X.Y.Z" in
the NCCL documentation.

---

## Issue 18: Missing `libdrm/drm.h` — `rocm_smi` dependency

**Error:**
```
/opt/rocm/include/rocm_smi/kfd_ioctl.h:26:10: fatal error: libdrm/drm.h: No such file or directory
```

**Root cause:** PyTorch's `intra_node_comm.cpp` (symmetric memory subsystem)
includes ROCm SMI headers which in turn include `kfd_ioctl.h`. That header
requires `libdrm/drm.h` from the `libdrm-dev` package, which was not installed.

**Fix:** Add `libdrm-dev` to the system packages in Stage 1:
```dockerfile
RUN apt-get update && apt-get install -y \
    ...
    libdrm-dev \
    ...
```
