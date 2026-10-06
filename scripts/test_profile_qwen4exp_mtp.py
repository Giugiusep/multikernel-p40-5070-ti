#!/usr/bin/env python3
"""Check replay accounting against a small deterministic trace."""

import importlib.util
import csv
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


MODULE = Path(__file__).with_name("profile-qwen4exp-mtp.py")
spec = importlib.util.spec_from_file_location("qwen_profile", MODULE)
assert spec and spec.loader
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def main() -> None:
    trace = """MTP_COMMIT slot=0 ordinal=1 token=10
MTP_DRAFT start_us=100 end_us=102 slots=1
MTP_DECODE start_us=103 end_us=203 tokens=2 output=1
MTP_ROUND slot=0 proposed=1 accepted=0 rejection=0 committed=1 restore=1 replay=0 end_us=205
MTP_VERIFY slot=0 pos=0 proposed=11 verified=12 candidate_logit=1 verified_logit=2
MTP_DECODE start_us=206 end_us=306 tokens=1 output=1
MTP_ROUND slot=0 proposed=1 accepted=1 rejection=-1 committed=2 restore=0 replay=1 end_us=308
MTP_VERIFY slot=0 pos=0 proposed=12 verified=12 candidate_logit=2 verified_logit=2
MTP_COMMIT slot=0 ordinal=2 token=12
MTP_COMMIT slot=0 ordinal=3 token=13
MTP_DRAFT start_us=400 end_us=403 slots=1
MTP_DECODE start_us=404 end_us=504 tokens=2 output=1
MTP_ROUND slot=0 proposed=1 accepted=1 rejection=-1 committed=2 restore=0 replay=0 end_us=506
MTP_VERIFY slot=0 pos=0 proposed=14 verified=14 candidate_logit=2 verified_logit=2
MTP_COMMIT slot=0 ordinal=4 token=14
MTP_COMMIT slot=0 ordinal=5 token=15
[ Prompt: 10.0 t/s | Generation: 5.0 t/s ]
"""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "trace.log"
        path.write_text(trace)
        result = profile.parse(path)
    assert result["tokens"] == 5
    assert result["initial"] == 1
    assert len(result["groups"]) == 2
    assert [len(group["passes"]) for group in result["groups"]] == [2, 1]
    assert [group["useful_commits"] for group in result["groups"]] == [2, 2]
    assert sum(group["passes"][0]["accepted"] for group in result["groups"]) == 1
    assert result["accepted_by_pos"][0] == [1, 2]
    assert [group["round_ms"] for group in result["groups"]] == [0.208, 0.106]

    base = """MTP_COMMIT slot=0 ordinal=1 token=10
MTP_DRAFT start_us=90 end_us=95 slots=1
MTP_DECODE start_us=100 end_us=300 tokens=2 output=1
MTP_ROUND slot=0 proposed=1 accepted=1 rejection=-1 committed=2 restore=0 replay=0 end_us=310
MTP_COMMIT slot=0 ordinal=2 token=11
MTP_COMMIT slot=0 ordinal=3 token=12
[ Prompt: 10.0 t/s | Generation: 5.0 t/s ]
"""
    events = "".join(
        f"ROUTE_TOPK t_us={120 + row * 48 + layer} layer={layer} row=0 experts=1,2,3,4,5,6\n"
        for row in range(2) for layer in range(48)
    )
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        baseline = directory / "baseline.log"
        routing = directory / "routing.log"
        output = directory / "routing.csv"
        baseline.write_text(base)
        routing.write_text(base.replace("MTP_ROUND", events + "MTP_ROUND"))
        subprocess.run([sys.executable, str(Path(__file__).with_name("analyze-qwen4exp-routing.py")),
                        str(baseline), str(routing), "--experts-per-row", "6", "--output", str(output)], check=True,
                       capture_output=True, text=True)
        with output.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["width"] == "2"
    assert rows[0]["unique_layer_experts"] == "288"
    assert rows[0]["selected_expert_slots"] == "576"
    assert rows[0]["reuse_factor"] == "2.0"
    print("replay and routing accounting passed")


if __name__ == "__main__":
    main()
