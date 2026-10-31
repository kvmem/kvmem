"""Read-only native Linux AMD board counters; these are NOT process VRAM."""
from pathlib import Path
import re
import time


def read_counter(path, required=False):
    attempts = 3 if required else 1
    for attempt in range(attempts):
        try:
            return int(path.read_text().strip()), None
        except (OSError, ValueError) as exc:
            if attempt + 1 == attempts:
                if required:
                    raise RuntimeError(f'Required GPU counter unavailable: {path}: {exc}') from exc
                return None, str(exc)
            time.sleep(0.05)


def discover_amd_cards(root=Path('/sys/class/drm')):
    cards = []
    for card in sorted(root.glob('card*')):
        if not re.fullmatch(r'card\d+', card.name):
            continue
        device = card / 'device'
        vendor = device / 'vendor'
        if vendor.exists() and vendor.read_text().strip().lower() == '0x1002':
            if (device / 'mem_info_vram_total').is_file():
                cards.append(device)
    return cards


def sample_amd_cards(cards):
    values = {}
    for device in cards:
        row = {}
        for field in ('mem_info_vram_total', 'mem_info_vram_used',
                      'mem_info_gtt_total', 'mem_info_gtt_used', 'gpu_busy_percent'):
            path = device / field
            value, error = read_counter(path, required=field in {
                'mem_info_vram_total', 'mem_info_vram_used'})
            if error is None:
                row[field] = value
            else:
                row.setdefault('unavailable', {})[field] = error
        values[str(device)] = row
    return values
