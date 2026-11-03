"""Radiance API -> console UI contract. All values come from the current worker."""
import copy

def effective_settings(info):
    native=info.get('request_defaults',{})
    effective={k:native[k] for k in ('temperature','top_p','top_k','min_p','reasoning_effort') if native.get(k) is not None}
    for source,target in [('max_tokens','max_tokens'),('thinking_token_budget','reasoning_budget_tokens')]:
        if source in native:effective[target]=-1 if native[source] is None else native[source]
    for key in ('enable_thinking','preserve_thinking','reasoning_effort'):
        if key in native.get('chat_template_kwargs',{}):effective[key]=native['chat_template_kwargs'][key]
    return effective

def enrich_props(props,info,snapshot):
    out=copy.deepcopy(props)
    out['kvmem']=copy.deepcopy(snapshot.get('kvmem',{}))
    effective=effective_settings(info)
    out['effective_settings']={'effective':effective,'source':'server_info.request_defaults'}
    out['kvmem']['defaults']=effective
    engine=info.get('engine',{})
    out['radiance_engine']=engine
    out['radiance_server_info']=info
    spec=engine.get('speculation',{})
    out['speculative']={'algorithm':spec.get('drafter','none') if spec.get('enabled') else 'none',
                        'draft_tokens':spec.get('depth',0)}
    policy=out['kvmem'].get('draft_policy')
    if policy in ('fixed','adaptive'):out['speculative']['adaptive']=policy=='adaptive'
    return out

def speculative_event(event):
    timings=event.get('timings') or {}
    offered,accepted=timings.get('draft_n'),timings.get('draft_n_accepted')
    if type(offered) is int and type(accepted) is int and 0<=accepted<=offered:
        event.setdefault('kvmem',{})['speculative']={'backend':'draft-mtp',
            'mtp_drafted_tokens':offered,'mtp_accepted_tokens':accepted}
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
