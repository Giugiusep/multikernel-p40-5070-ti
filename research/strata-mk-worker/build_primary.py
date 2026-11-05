"""Relink isolated adapter against unchanged, fingerprinted RTX reference objects."""
from pathlib import Path
import os
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "Strata-mk-worker"
REFERENCE = ROOT.parent / "Strata/build"
OUT = Path(os.environ.get("STRATA_MK_BUILD_DIR", str(ROOT / "primary-build")))
OUT.mkdir(exist_ok=True)
source = SOURCE / "src/core/expert_source.cpp"
obj = OUT / "expert_source.cpp.o"
cmd = ["/usr/bin/g++-13", "-O3", "-DNDEBUG", "-std=c++20", "-Wall", "-Wextra", "-DSTRATA_NATIVE_EXPERTS=1",
       "-I" + str(SOURCE / "include"),
       "-I" + str(SOURCE / "third_party/llama.cpp/ggml/include"),
       "-isystem", "/usr/local/cuda-12.8/targets/x86_64-linux/include", "-c", str(source), "-o", str(obj)]
subprocess.run(cmd, check=True)
library = OUT / "libstrata_engine.a"
shutil.copy2(REFERENCE / "libstrata_engine.a", library)
subprocess.run(["ar", "r", str(library), str(obj)], check=True)
libs = ["libstrata_engine.a", "libstrata_prefill.a", "libstrata_spec.a", "libstrata_engine.a",
        "libstrata_kernels.a", "libstrata_core.a", "libstrata_kernels_cpu.a", "ggml/src/libggml-cpu.a", "ggml/src/libggml-base.a"]
paths = [library if name == "libstrata_engine.a" else REFERENCE / name for name in libs]
generate = REFERENCE / "CMakeFiles/strata.dir/src/program/generate.cpp.o"
toolkit = "/usr/local/cuda-12.8/targets/x86_64-linux/lib"
link = ["/usr/bin/g++-13", "-O3", "-DNDEBUG", str(generate), "-o", str(OUT / "strata"),
        "-L" + toolkit, "-Wl,-rpath," + toolkit, *map(str, paths),
        "-lm", "-lcudart", "-ldl", "-lcublas", "-lcublasLt", "-lculibos", "-lcudadevrt", "-lcudart_static", "-lrt", "-lpthread", "-ldl"]
subprocess.run(link, check=True)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {"source_head": subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
            "build_method": "replace only expert_source.cpp.o in copied reference static archive; retain reference GPU kernels and driver",
            "reference_objects": {str(p): sha(p) for p in set(paths + [generate]) if p != library},
            "original_engine_archive_sha256": sha(REFERENCE / "libstrata_engine.a"),
            "adapter_source_sha256": sha(source), "adapter_header_sha256": sha(source.parent / "mk_expert_client.hpp"),
            "binary_sha256": sha(OUT / "strata"), "compile": cmd, "link": link}
(OUT / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Isolated primary binary:", OUT / "strata")
