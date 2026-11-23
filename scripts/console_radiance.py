"""Radiance runtime, frozen launch identities and console telemetry -> UI contract."""
import copy
import re

def effective_settings(info, template_effort=None):
    native=info.get('request_defaults',{})
    effective={k:native[k] for k in ('temperature','top_p','top_k','min_p','reasoning_effort') if native.get(k) is not None}
    for source,target in [('max_tokens','max_tokens'),('thinking_token_budget','reasoning_budget_tokens')]:
        if source in native:effective[target]=-1 if native[source] is None else native[source]
    for key in ('enable_thinking','preserve_thinking','reasoning_effort'):
        if key in native.get('chat_template_kwargs',{}):effective[key]=native['chat_template_kwargs'][key]
    effort = effective.get('reasoning_effort')
    if not effort and template_effort and template_effort in (info.get('capabilities', {}).get('reasoning_efforts') or []):
        effort = effective['reasoning_effort'] = template_effort
    if effort in ('none', 'minimal'):
        effective['enable_thinking'] = False
    elif effort and info.get('capabilities', {}).get('reasoning'):
        effective.setdefault('enable_thinking', True)
    return effective

def monitor_execution(engine, launch_devices=None):
    """Translate a complete worker rank list; startup free memory is not telemetry."""
    ranks = engine.get('devices')
    count = engine.get('tensor_parallel')
    if type(count) is not int or count < 1 or not isinstance(ranks, list) or len(ranks) != count:
        return None
    devices = []
    for row in ranks:
        if not isinstance(row, dict):
            return None
        rank, ordinal, name = row.get('rank'), row.get('device'), row.get('name')
        if type(rank) is not int or type(ordinal) is not int or ordinal < 0 or not isinstance(name, str) or not name.strip():
            return None
        device = {'rank': rank, 'hip_ordinal': ordinal, 'name': name.strip()}
        for key in ('arch', 'uuid', 'pci_bus_id'):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                device[key] = value.strip()
        # The launch plan records the physical devices behind the HIP visibility mask.
        # Require the worker's complete rank list AND the exact frozen launch mapping.
        bindings = launch_devices if isinstance(launch_devices, list) and len(launch_devices) == count else []
        matches = [b for b in bindings if isinstance(b, dict) and b.get('rank') == rank
                   and b.get('hip_ordinal') == ordinal and b.get('name', '').strip() == name.strip()]
        if len(matches) == 1:
            binding = matches[0]
            uuid, pci = binding.get('uuid'), binding.get('pci_bus_id')
            if (isinstance(uuid, str) and re.fullmatch(r'GPU-[0-9a-fA-F]{32}', uuid)
                    and isinstance(pci, str) and re.fullmatch(r'[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', pci)
                    and ('uuid' not in device or device['uuid'].lower() == uuid.lower())
                    and ('pci_bus_id' not in device or device['pci_bus_id'].lower() == pci.lower())):
                device.setdefault('uuid', uuid)
                device.setdefault('pci_bus_id', pci)
        capacity = row.get('vram_bytes')
        if type(capacity) is int and capacity > 0:
            device['total_bytes'] = capacity
        devices.append(device)
    devices.sort(key=lambda d: d['rank'])
    if [d['rank'] for d in devices] != list(range(count)) or len({d['hip_ordinal'] for d in devices}) != count:
        return None
    return {'mode': 'single' if count == 1 else 'tensor', 'monitor_devices': devices}

def enrich_props(props,info,snapshot,template_effort=None,launch_devices=None):
    out=copy.deepcopy(props)
    out['kvmem']=copy.deepcopy(snapshot.get('kvmem',{}))
    effective=effective_settings(info,template_effort)
    out['effective_settings']={'effective':effective,'source':'radiance-worker'}
    out['kvmem']['defaults']=effective
    engine=info.get('engine') or {}
    execution = out.get('execution') or {}
    if not isinstance(execution.get('monitor_devices'), list):
        reported = monitor_execution(engine, launch_devices)
        if reported:
            out['execution'] = {**execution, **reported}
    out['radiance_engine']=engine
    out['radiance_server_info']=info
    spec=engine.get('speculation',{})
    out['speculative']={'algorithm':spec.get('drafter','none') if spec.get('enabled') else 'none',
                        'draft_tokens':spec.get('depth',0)}
    policy=out['kvmem'].get('draft_policy')
    if policy in ('fixed','adaptive'):out['speculative']['adaptive']=policy=='adaptive'
    return out


def enrich_hardware(snapshot, hardware):
    """Supply console hardware only where Radiance has no native hardware sampler."""
    out = copy.deepcopy(snapshot)
    if out.get('backend') == 'radiance' and not isinstance(out.get('hardware'), dict):
        out['hardware'] = copy.deepcopy(hardware)
    return out

def speculative_event(event):
    timings=event.get('timings') or {}
    offered,accepted=timings.get('draft_n'),timings.get('draft_n_accepted')
    if type(offered) is int and type(accepted) is int and 0<=accepted<=offered:
        if offered:
            event.setdefault('kvmem',{})['speculative']={'backend':'radiance',
                'drafted_tokens':offered,'accepted_tokens':accepted}
    return event

def enrich_status(state,snapshot):
    out=copy.deepcopy(state)
    actual=snapshot.get('kvmem',{}).get('scoring')
    if not isinstance(actual,dict):return out
    planned=state.get('planned_scoring') or {}
    out['effective_scoring']=copy.deepcopy(actual)
    complete=bool(planned) and all(key in actual for key in planned)
    out['scoring_confirmed']=complete and all(actual[key]==value for key,value in planned.items())
    out['scoring_mismatch']=complete and not out['scoring_confirmed']
    out['scoring_source']='radiance-monitor'
    return out
