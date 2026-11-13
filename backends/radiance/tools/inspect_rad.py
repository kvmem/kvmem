#!/usr/bin/env python3
"""Read-only .rad v2 directory accounting; never loads weights onto a GPU.

The directory schema follows upstream abi/rad_format.h (RAD_MAX_RANK=6).
File bytes are not a runtime VRAM estimate: plugins can omit weights, repack
encodings or place embeddings on host memory. Incomplete downloads are refused.
"""
import argparse
import collections
import hashlib
import json
import struct
from pathlib import Path


def inspect(path, allow_partial=False, include_entries=False, max_directory_bytes=128*1024*1024,
            include_vocab=False):
    header = struct.Struct("<II30Q")
    entry = struct.Struct("<3Q2I6q3iI5Q")
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(header.size)
        if len(raw) != header.size:
            raise ValueError("incomplete header")
        h = header.unpack(raw)
        if h[0] != 0x31444152 or h[1] != 2 or h[2] < size or (h[2] != size and not allow_partial):
            raise ValueError("not a complete .rad v2 container")

        def read_at(offset, count):
            if count > max_directory_bytes or offset > size or count > size - offset:
                raise ValueError("directory range outside inspection limits")
            stream.seek(offset)
            data = stream.read(count)
            if len(data) != count:
                raise ValueError("truncated directory")
            return data

        strings = read_at(h[8], h[9])

        def string(index):
            if not 0 <= index < len(strings):
                raise ValueError("invalid string reference")
            end = strings.find(b"\0", index)
            if end < index:
                raise ValueError("invalid string reference")
            return strings[index:end].decode("utf-8")

        # Read the original declaration, not the filename: a DFlash container
        # can retain NextN metadata even though conversion omitted MTP weights.
        metadata = {}
        vision_metadata = {}
        for offset, kind, _, value in struct.iter_unpack("<QIIQ", read_at(h[10], h[11] * 24)):
            key = string(offset)
            if key.startswith(("draft.", "vision_config.")) or key in ("qwen35.nextn_predict_layers", "mtp_num_hidden_layers"):
                target = vision_metadata if key.startswith("vision_config.") else metadata
                if kind == 1:
                    target[key] = value if value < 2**63 else value - 2**64
                elif kind == 2:
                    target[key] = string(value)
        counts = collections.Counter()
        directory = read_at(h[12], h[13] * entry.size)
        encodings = read_at(h[16], h[17] * 568) if include_entries else b""
        planes = read_at(h[14], h[15] * 16) if include_entries else b""
        totals = collections.defaultdict(int)
        rows = []
        for e in entry.iter_unpack(directory):
            name = string(e[0])
            if not 0 <= e[4] <= 6 or e[16] > h[2] or e[17] > h[2] - e[16]:
                raise ValueError("invalid weight entry")
            group = name.split(".")[0]
            totals[group] += e[17]
            counts[group] += 1
            rows.append({"name": name, "bytes": e[17], "shape": e[5:5 + e[4]],
                         "encoding_index": e[3], "layer": e[11], "offset": e[16]})
            if include_entries:
                if e[3] >= h[17] or e[15] + e[14] > h[15]:
                    raise ValueError("weight encoding/plane index out of range")
                rows[-1]["encoding_sha256"] = hashlib.sha256(encodings[e[3]*568:(e[3]+1)*568]).hexdigest()
                row_planes = list(struct.iter_unpack("<QQ", planes[e[15]*16:(e[15]+e[14])*16]))
                if any(off < e[16] or off-e[16]+count > e[17] for off, count in row_planes):
                    raise ValueError("plane outside weight payload")
                rows[-1]["relative_planes"] = [(off-e[16], count) for off, count in row_planes]
        result = {"path": str(path), "file_bytes": size, "declared_file_bytes": h[2],
                "complete_container": h[2] == size, "arch_id": string(h[3]),
                "model": string(h[4]), "quant": string(h[5]), "entries": len(rows),
                "draft_metadata": metadata, "vision_metadata": vision_metadata, "entries_by_name_prefix": dict(counts),
                "directory_weight_bytes": sum(totals.values()), "bytes_by_name_prefix": dict(totals),
                "largest_weights": sorted(rows, key=lambda x: x["bytes"], reverse=True)[:15],
                "note": "container directory only; not runtime VRAM or a checksum verification"}
        if include_entries:
            result["weight_entries"] = rows
        if include_vocab:
            # abi/rad_format.h: RadVocabHeader. Inspect the header and array
            # extents only; vocabulary/token payloads and weights stay unread.
            vocab_header = struct.Struct("<4I5Q6i2I5Q")
            vocab = None
            voff, vbytes = h[18:20]
            if voff > size or vbytes > size - voff:
                raise ValueError("vocab section outside file")
            if vbytes:
                if vbytes < vocab_header.size or voff % 8:
                    raise ValueError("invalid vocab header size or alignment")
                v = vocab_header.unpack(read_at(voff, vocab_header.size))
                nt, nm, ns = v[1:4]
                for off, count, stride, alignment, optional in (
                        (v[4], nt, 8, 8, False), (v[5], nt, 4, 4, True),
                        (v[6], nt, 1, 1, True), (v[7], nm, 8, 4, False),
                        (v[8], ns, 32, 8, False)):
                    if count == 0 or (off == 0 and optional):
                        continue
                    if (not off or off < voff or off - voff > vbytes
                            or count * stride > vbytes - (off - voff) or off % alignment):
                        raise ValueError("vocab array outside section or misaligned")
                template = string(v[17]).encode('utf-8')
                vocab = dict(kind=v[0], kind_name={1:'BPE',2:'Unigram',3:'WordPiece',4:'RWKV'}.get(v[0],str(v[0])),
                             tokens=nt, merges=nm, steps=ns, bytes=vbytes,
                             chat_template_bytes=len(template),
                             chat_template_sha256=hashlib.sha256(template).hexdigest() if template else None)
            result['vocab'] = vocab
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--allow-partial-directory", action="store_true",
                        help="inspect already downloaded metadata only; does NOT verify or qualify the weights for loading")
    args = parser.parse_args()
    print(json.dumps(inspect(args.model, args.allow_partial_directory), indent=2))
