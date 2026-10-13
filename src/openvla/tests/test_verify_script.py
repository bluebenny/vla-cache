import json
import os

from helpers import SEQ_LEN, load_module_from_path

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "vla_cache_scripts", "verify_vprime_equivalence.py")


def test_verify_script_passes_on_tiny_model(tmp_path):
    verify = load_module_from_path("verify_vprime_equivalence", SCRIPT)
    output = tmp_path / "report.json"
    assert verify.main(["--tiny_random", "--output", str(output)]) == 0

    report = json.loads(output.read_text())
    assert report["A_logic"]["num_reusable"] > 0
    # The compared step really reused tokens: layers 0-1 see the full prompt, later layers fewer
    tokens = report["B_in_module"]["tokens_per_layer"]
    assert tokens[0] == tokens[1] == SEQ_LEN and min(tokens[2:]) < SEQ_LEN
    assert report["C_sdpa_info"]["tokens_per_layer_v"] == tokens
