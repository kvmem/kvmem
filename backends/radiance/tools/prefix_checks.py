"""Cross-check real input checkpoints, native resumes and API cache accounting."""
import re


def input_prefix_continuations(log_lines, rounds):
    def records(marker):
        return [dict(re.findall(r"(\w+)=([^ ]+)", line)) for line in log_lines if marker + " " in line]
    retained = records("KVMEM_INPUT_RETAIN")
    resumes = records("KVMEM_PREFIX_RESUME")
    selected = records("KVMEM_CHECKPOINT_SELECT")
    ready = records("KVMEM_INPUT_READY")
    by_request = {item["request"]: item for item in retained}
    ready_ids = {item["request"] for item in ready}
    valid = len(rounds) > 1 and len(resumes) == len(rounds)-1
    valid = valid and len(selected) == len(resumes)
    # The dedicated fixture starts with enough tool history for a real input
    # checkpoint; unlike Task 2's short seed, every follow-up is expected to hit.
    valid = valid and len(retained) == len(rounds)
    for index, (resume, choice, row) in enumerate(zip(resumes, selected, rounds[1:]), 1):
        cached = (row.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        previous = retained[index-1] if index <= len(retained) else {}
        frontier = int(resume.get("frontier", 0))
        valid = valid and resume.get("exact_prefix") == "1" and choice.get("purpose") == "input"
        valid = valid and resume.get("request") == choice.get("request")
        valid = valid and 0 < frontier == cached == int(choice.get("frontier", 0))
        valid = valid and frontier == int(previous.get("frontier", 0)) <= rounds[index-1]["usage"]["prompt_tokens"]
        valid = valid and frontier <= int(choice.get("lcp", 0))
        valid = valid and previous.get("request") in ready_ids and previous.get("request") in by_request
    # Tool continuation must not inherit the previous user's cold-history Query.
    spans = records("KVMEM_QUERY_SPAN")
    valid = valid and len(spans) == len(rounds) and all(
        x.get("proven") == "0" and x.get("begin") == x.get("end") == "-1" for x in spans)
    return {"pass": bool(valid), "retained": retained, "resumes": resumes, "selected": selected}
