"""GPU HTTP regression for switching back to a sparse, unaligned checkpoint."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--gpu', default=os.environ.get('KVMEM_TEST_GPU'), help='GPU UUID')
    parser.add_argument('--gpu-layers', type=int, default=99)
    parser.add_argument('--expect-mixed-capture', action='store_true')
    parser.add_argument('--mtp', action='store_true')
    parser.add_argument('--dflash2-draft', type=Path)
    parser.add_argument('--mtp-state', choices=['replay', 'snapshots'], default='snapshots')
    parser.add_argument('--block-tokens', type=int, choices=[32, 64, 128], default=64)
    parser.add_argument('--turns', type=int, default=4)
    parser.add_argument('--cancel-rounds', type=int, default=1)
    parser.add_argument('--checkpoint-policy', choices=['legacy', 'spread'], default='legacy')
    parser.add_argument('--checkpoint-min-gap', type=int, help='override nearby history spacing; zero disables thinning')
    parser.add_argument('--checkpoint-disk-mib', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.model is None:
        name = 'KVMEM_TEST_MTP_MODEL' if args.mtp else 'KVMEM_TEST_MODEL'
        args.model = Path(os.environ[name]) if os.environ.get(name) else None
    if not args.model or not args.model.is_file() or not args.exe.is_file() or not args.gpu:
        return 77
    occupancy = subprocess.check_output(['nvidia-smi', '-i', args.gpu,
        '--query-gpu=uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    if int(occupancy.split(',')[2]) > 512 or int(occupancy.split(',')[3]) > 5:
        raise RuntimeError('requested GPU is occupied; regression not started')
    args.out.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [str(args.exe), '-m', str(args.model), '--host', '127.0.0.1', '--port', str(port),
               '-c', '8192', '-ngl', str(args.gpu_layers), '-b', '128', '-ub', '128', '--no-ui', '--no-think',
               '--kvmem', '--kvmem-budget', '512', '--kvmem-gen-reserve', '128',
               '--kvmem-block-tokens', str(args.block_tokens), '--kvmem-sessions', '4', '--kv-dtype', 'q8_0',
               '--spec-type', 'draft-dflash' if args.dflash2_draft else 'draft-mtp' if args.mtp else 'none', '--kvmem-trace']
    if args.mtp or args.dflash2_draft:
        command += ['--spec-draft-n-max', '5' if args.dflash2_draft else '3', '--spec-kv-dtype', 'f16', '--kvmem-mtp-state', args.mtp_state]
    if args.dflash2_draft:
        command += ['--spec-draft-model', str(args.dflash2_draft)]
    if args.checkpoint_disk_mib:
        command += ['--kvmem-disk-mib', str(args.checkpoint_disk_mib), '--kvmem-disk-path', str(args.out / 'spill')]
    env = {k: v for k, v in os.environ.items() if not k.startswith(('KVMEM_', 'NINFER_', 'GGML_', 'LLAMA_ARG_'))}
    env.pop('CUDA_LAUNCH_BLOCKING', None)
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID', GGML_DISABLE_VULKAN='1')
    env.update(KVMEM_CHECKPOINT_POLICY=args.checkpoint_policy, KVMEM_CHECKPOINT_INTERVAL='512',
               KVMEM_CHECKPOINT_COUNT='12', KVMEM_CHECKPOINT_HOT='2', KVMEM_CHECKPOINT_HOST_MIB='1536')
    if args.checkpoint_min_gap is not None:env['KVMEM_CHECKPOINT_MIN_GAP']=str(args.checkpoint_min_gap)
    env['PATH'] = str(args.exe.parent) + os.pathsep + env['PATH']
    (args.out / 'manifest.json').write_text(json.dumps({'command': command, 'gpu_uuid': args.gpu,
        'occupancy': occupancy, 'policy_env': {k:v for k,v in env.items() if k.startswith('KVMEM_')}}, indent=2))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f'http://127.0.0.1:{port}'

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(base + path, data=data, headers={'Content-Type': 'application/json'})
        with opener.open(request, timeout=3 if body is None else 180) as response:
            return json.load(response)

    rows = []
    exchanges = []
    log_path = args.out / 'server.log'
    with log_path.open('wb') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=log,
                                 creationflags=0x08000000 if os.name == 'nt' else 0)
        try:
            deadline = time.monotonic() + 120
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f'server exited {child.returncode}')
                try:
                    call('/health')
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('server readiness')
                    time.sleep(.25)

            def chat(name, messages, limit):
                body = {'messages': messages, 'max_tokens': limit,
                        'temperature': 0, 'seed': 42, 'enable_thinking': False}
                response = call('/v1/chat/completions', body)
                exchanges.append(json.loads(json.dumps({'case':name, 'request':body, 'response':response})))
                (args.out / 'exchanges.json').write_text(json.dumps(exchanges, indent=2), encoding='utf-8')
                assert child.poll() is None
                assert response['usage']['completion_tokens'] > 0, response
                row = {'case': name, **response['usage']}
                rows.append(row)
                (args.out / 'results.json').write_text(json.dumps(rows, indent=2))
                print(json.dumps(row), flush=True)
                return response

            short = [{'role': 'user', 'content': 'Reply briefly with one fact about the Moon.'}]
            history = [{'role': 'system', 'content': 'Answer briefly.'}, {'role': 'user', 'content':
                       'The violet observatory studies pulsars and cosmic rays. ' * 160 + 'State one fact.'}]
            chat('short', short, 16)
            assert chat('long', history, 16)['usage']['prompt_tokens'] > 640
            chat('switch-away', short, 128)
            chat('repeat-short', short, 128)
            questions = ['What causes day and night on Earth?', 'Why does the Moon appear to change shape?',
                'What is the Sun made of?', 'Why does Mars look red?', 'What are Saturn\'s rings made of?',
                'How do telescopes collect light?', 'What is a comet\'s tail made of?',
                'Why do stars appear to twinkle?', 'What keeps planets moving around the Sun?',
                'How does an astronaut breathe in space?', 'What is the Milky Way?']
            for turn in range(args.turns):
                question = (f'Turn {turn}: state one brief fact.' if turn == 0 else
                    f'New question {turn}: {questions[(turn-1) % len(questions)]} Reply with one complete sentence.')
                history.append({'role': 'user', 'content': question})
                response = chat(f'restore-long-{turn}', history, 8 if turn == 0 else 48)
                assert response['usage']['prompt_cache_hit_tokens'] > 0, 'restored history was recomputed in full'
                history.append({'role': 'assistant', 'content': response['choices'][0]['message']['content'] or 'Recorded.'})
                chat(f'next-short-{turn}', short, 16)
            trace = log_path.read_text(encoding='utf-8', errors='replace')
            if args.expect_mixed_capture:
                assert 'harvest_sync reason=non_uniform_capture' in trace, 'mixed capture path was not exercised'
            restored = re.findall(r'checkpoint_tail_restore row=(\d+) block=(\d+)', trace)
            reuse = [int(x) for x in re.findall(r'prefix_hit_rows=(\d+)', trace)]
            staged = re.findall(r'stage_in_packed block=(\d+) slot=\d+ n=(\d+) orig=(\d+)', trace)
            full_blocks = {int(block) for block, n, _ in staged if int(n) == args.block_tokens}
            partial_blocks = {row // args.block_tokens for row in reuse
                              if row > 128 and row % args.block_tokens}
            if args.checkpoint_policy == 'legacy':
                # v0.9.0 can stage the partial block at full width from the
                # packed store before restoring a checkpoint inside it, which
                # materializes the retained prefix without a cold-tail reload.
                # Accept either the transaction or that full-width staging.
                cold = {int(block) for row, block in restored if int(row) % args.block_tokens}
                assert cold or (partial_blocks & full_blocks), 'cold partial block was not restored'
            else:
                # Spread may restore a different native frontier whose partial
                # block is already resident, so no tail reload is required.
                assert partial_blocks, 'unaligned history restore was not exercised'
            assert not re.search(r'position/coverage gap|checkpoint_restore_fallback|multimodal_error|CUDA error:', trace)
            if args.mtp or args.dflash2_draft:
                assert sum(map(int, re.findall(r'spec_stats[^\r\n]* n_drafted=(\d+)', trace))) > 0

            def log_suffix(offset):
                with log_path.open('rb') as reader:
                    reader.seek(offset)
                    return reader.read().decode(errors='replace')

            cancel_checks = []
            for cancel_round in range(args.cancel_rounds):
                cancelled = history + [{'role': 'user', 'content':
                             f'Cancellation round {cancel_round}. ' + 'Additional archive statement. ' * 600}]
                offset = log_path.stat().st_size
                request = urllib.request.Request(base + '/v1/chat/completions',
                    data=json.dumps({'messages': cancelled, 'max_tokens': 128, 'stream': True,
                                     'temperature': 0, 'enable_thinking': False}).encode(),
                    headers={'Content-Type': 'application/json'})
                response = opener.open(request, timeout=180)
                try:
                    assert response.readline().startswith(b'data: '), 'stream did not open'
                    deadline = time.monotonic() + 30
                    while True:
                        pending = log_suffix(offset)
                        if re.search(r'multimodal_compute rows=128 .*replay=0', pending):
                            assert 'KVMEM_CHAT_PREFILL' not in pending, 'prefill completed before cancellation'
                            break
                        assert time.monotonic() < deadline, 'prefill did not start'
                        time.sleep(.01)
                finally:
                    response.close()
                deadline = time.monotonic() + 30
                while True:
                    pending = log_suffix(offset)
                    if 'KVMEM_CHECKPOINT_ROLLBACK' in pending:
                        break
                    assert time.monotonic() < deadline, 'cancelled request did not roll back'
                    time.sleep(.02)
                chat(f'after-cancel-{cancel_round}-short', short, 16)
                recovered = chat(f'after-cancel-{cancel_round}-long', history, 8)
                hit_tokens = recovered['usage']['prompt_cache_hit_tokens']
                assert hit_tokens > 0, 'cancellation discarded the retained long prefix'
                cancel_checks.append({'round':cancel_round, 'rolled_back':True,
                    'recovery_mode':'preserved', 'next_requests_completed':True,
                    'recovered_prefix_tokens':recovered['usage']['prompt_cache_hit_tokens']})
            trace = log_path.read_text(encoding='utf-8', errors='replace')
            assert not re.search(r'position/coverage gap|CUDA error:|checkpoint_restore_fallback|KVMEM cache clear failed', trace)
            assert 'multimodal_rollback_failed' not in trace
            restored = re.findall(r'checkpoint_tail_restore row=(\d+) block=(\d+)', trace)
            if args.dflash2_draft:
                spec = call('/props')['speculative']
                assert spec['schema_version'] == 1 and spec['algorithm'] == 'dflash2', spec
                stats = spec['statistics']
                assert stats['requests'] >= len(rows) and stats['rounds'] > 0, stats
                assert 0 <= stats['accepted_tokens'] <= stats['drafted_tokens'] and stats['drafted_tokens'] > 0, stats
                (args.out / 'speculative.json').write_text(json.dumps(spec, indent=2))
            (args.out / 'summary.json').write_text(json.dumps({'pass': True, 'requests': len(rows),
                'checkpoint_tail_restores': restored, 'cancel_recovery': True,
                'gpu_uuid': args.gpu, 'mtp': args.mtp, 'dflash2': bool(args.dflash2_draft),
                'mtp_state':args.mtp_state if args.mtp or args.dflash2_draft else None,
                'block_tokens':args.block_tokens, 'turns':args.turns,
                'cancel_checks':cancel_checks,
                'recovery_policy':'preserve_only'}, indent=2))
        finally:
            child.terminate()
            try:
                child.wait(15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
