#!/usr/bin/env python3
"""One real image request plus one retained-history follow-up."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request



CODE = "CODE-7319"


def make_image() -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGB", (1024, 512), "white")
    draw = ImageDraw.Draw(image)
    fonts = [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / name
             for name in ("arialbd.ttf", "calibrib.ttf", "segoeuib.ttf")]
    font = next((ImageFont.truetype(str(path), 128) for path in fonts if path.is_file()), ImageFont.load_default())
    draw.text((120, 165), CODE, fill="black", font=font)
    data = io.BytesIO()
    image.save(data, "PNG")
    return data.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "worker", "vision-worker", "mmproj", "model", "tokenizer", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, default=18286)
    parser.add_argument("--mtp-drafts", type=int, choices=range(4), default=2)
    parser.add_argument("--kv-type",choices=("f16","q8_0"),default="f16")
    parser.add_argument("--vision-device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--image",type=Path,help="existing CODE-7319 PNG fixture; avoids the optional Pillow dependency")
    args = parser.parse_args()
    for name in ("source", "worker", "vision_worker", "mmproj", "model", "tokenizer", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    png = args.image.read_bytes() if args.image else make_image()
    (args.out / "code.png").write_bytes(png)
    image_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    command = [sys.executable, str(Path(__file__).with_name("server.py")),
               "--source", str(args.source), "--worker", str(args.worker), "--model", str(args.model),
               "--tokenizer", str(args.tokenizer), "--gpu", args.gpu, "--port", str(args.port),
               "--context", "16384", "--budget", "2048", "--reserve", "1024", "--prefill", "1024",
               "--max-tokens", "32", "--host-mib", "1024", "--history-reuse", "--mtp-drafts", str(args.mtp_drafts),
               "--vision-worker", str(args.vision_worker), "--mmproj", str(args.mmproj),
               "--vision-device", args.vision_device,"--kv-type",args.kv_type, "--vision-tokens", "300", "--vision-threads", "8",
               "--log",str(args.out/'native.log'),
               "--", "--threads", "4", "--stage-mib", "256", "--tail-skip", "0"]
    base = f"http://127.0.0.1:{args.port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, payload=None, timeout=600):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=timeout) as response:
            return json.load(response)

    first_body = {"model": "kvmem-polystrata-qwen35", "temperature": 0, "max_tokens": 32,
                  "chat_template_kwargs": {"enable_thinking": False},
                  "messages": [{"role": "user", "content": [
                      {"type": "image_url", "image_url": {"url": image_url}},
                      {"type": "text", "text": "Read the large black code. Reply with only the exact code."},
                  ]}]}
    result = {"status": "running", "command": command, "gpu_uuid": args.gpu, "expected": CODE}
    with (args.out / "server.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=0x08000000 if os.name == "nt" else 0)
        try:
            deadline = time.monotonic() + 600
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"frontend exited: {proc.returncode}")
                try:
                    request("/health", timeout=5)
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() > deadline:
                        raise TimeoutError("frontend health timeout")
                    time.sleep(.5)
            props = request("/props")
            assert props["modalities"]["vision"] is True, props
            assert props["kvmem"]["vision"] is True, props
            first = request("/v1/chat/completions", first_body)
            result.update(first_request=first_body,first=first)
            answer = first["choices"][0]["message"]["content"].strip()
            assert answer == CODE, first
            second_body = dict(first_body)
            second_body["messages"] = first_body["messages"] + [
                {"role": "assistant", "content": CODE},
                {"role": "user", "content": "Repeat only the exact code from the image."},
            ]
            second = request("/v1/chat/completions", second_body)
            result.update(second_request=second_body,second=second)
            repeated = second["choices"][0]["message"]["content"].strip()
            assert repeated == CODE, second
            cached = second["usage"]["prompt_tokens_details"]["cached_tokens"]
            assert cached > 0, second
            result.update(status="pass", props=props, first=first, second=second, cached_tokens=cached)
        except Exception as error:
            result.update(status="fail", error=str(error))
            raise
        finally:
            if proc.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=True)
                else:
                    proc.terminate()
                proc.wait(timeout=30)
            (args.out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "answer": CODE, "cached_tokens": result["cached_tokens"]}))


if __name__ == "__main__":
    main()
