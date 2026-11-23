"""Fixed-route chat transport and UI contract for the owned local service.

No prompts or responses are retained here. Only bounded request metadata lives
in memory. Cancellation closes the upstream connection, including during prefill.
"""
import copy
import http.client
import json
import math
import queue
import select
import socket
import threading
import time
from urllib.parse import urlsplit
from console_radiance import effective_settings, speculative_event

MAX_BODY = 64 * 1024 * 1024
PARAMS = ('temperature', 'top_p', 'top_k', 'min_p', 'seed', 'max_tokens',
          'reasoning_budget_tokens', 'reasoning_effort', 'enable_thinking', 'preserve_thinking')


def request_metrics(record, snapshot, row, backend):
    """Normalize exact request data for the chat's live/mean display contract."""
    finite = lambda value: type(value) in (int, float) and math.isfinite(value) and value >= 0
    metrics = dict(row) if row else {}
    if row:
        metrics['decode_tps_mean'] = row.get('decode_tps_mean', row.get('decode_tps'))
        # decode_tps on C++ monitors is a cumulative mean, not a live rate.
        live = None
        active = row in snapshot.get('active', []) and row.get('status') == 'generating'
        if active and backend in ('llamacpp','ninfer','radiance'):
            rate, window, sampled = (row.get(k) for k in
                                     ('decode_tps_live','decode_tps_window_ms','decode_tps_sampled_at_ms'))
            now = time.time() * 1000
            if all(finite(v) for v in (rate,window,sampled)) and window > 0:
                age = now - sampled
                if -1000 <= age <= max(6000,window*2)+1000:
                    live = rate
        elif active and backend == 'polystrata' and finite(row.get('decode_tps')):
            live = row['decode_tps']
        metrics['decode_tps'] = live
    # SSE timings are already bound to this browser request. They also work when
    # an engine has no monitor response-ID mapping. Never use the global 5 s rate.
    timings = record.get('timings') or {}
    for source, target in {
        'prompt_ms':'prefill_ms', 'prompt_n':'prompt_new_tokens', 'cache_n':'cached_tokens',
        'prompt_per_second':'prefill_tps', 'predicted_n':'generated_tokens',
        'predicted_ms':'decode_ms', 'predicted_per_second':'decode_tps_mean',
        'draft_n':'drafts_offered', 'draft_n_accepted':'drafts_accepted',
    }.items():
        if finite(timings.get(source)):
            metrics[target] = timings[source]
    if record.get('finish_reason'):
        metrics['finish_reason'] = record['finish_reason']
    if record.get('state') in ('completed','cancelled','error'):
        metrics['decode_tps'] = None
    return metrics


def context_payload(state, props, settings=None):
    launch = (state.get('active_config') or {}).get('parameters', {})
    backend = launch.get('Backend')
    native = props.get('kvmem', {})
    generation = props.get('default_generation_settings', {})
    params = generation.get('params', {})
    defaults = native.get('defaults', {})
    effective = {k: params[k] for k in PARAMS if k in params}
    sources = {k: 'engine' for k in effective}
    def put(key, value, source='engine'):
        if value is not None:
            effective[key] = value
            sources[key] = source
    put('max_tokens', params.get('max_tokens', params.get('n_predict')))
    for key in ('enable_thinking', 'reasoning_budget_tokens', 'preserve_thinking'):
        put(key, defaults.get(key))
    put('reasoning_effort', defaults.get('chat_template_kwargs', {}).get('reasoning_effort'))
    put('preserve_thinking', defaults.get('chat_template_kwargs', {}).get('preserve_thinking'))
    # These are explicit parameters of the running process, never the editable draft.
    for key, field in [('MaxTokens','max_tokens'), ('ThinkingBudget','reasoning_budget_tokens'),
                       ('ReasoningEffort','reasoning_effort'), ('PreserveThinking','preserve_thinking')]:
        if field not in effective:
            value = launch.get(key)
            # The Radiance launcher uses 0 to omit its native budget flag;
            # the native request protocol uses -1 for no bound and 0 to close
            # reasoning immediately. Report the actual startup meaning.
            if backend == 'radiance' and key == 'ThinkingBudget' and value == 0: value = -1
            if backend == 'ninfer' and key == 'PreserveThinking' and value is False: value = None
            put(field, value, 'launch')
    if backend == 'radiance' and settings:
        for key,value in effective_settings(settings,state.get('radiance_template_effort')).items(): put(key,value)
    mode = launch.get('ThinkingMode')
    if 'enable_thinking' not in effective and mode in ('on','off'):
        put('enable_thinking', mode == 'on', 'launch')
    if backend == 'ninfer':
        # NInfer's ordinary Qwen templates default to thinking; Gemma4 defaults
        # to final-only. An arbitrary custom template must report its own policy.
        model = str(props.get('model_alias', '')) + str(launch.get('Model', ''))
        if not launch.get('ChatTemplateFile') and 'enable_thinking' not in effective:
            if 'qwen' in model.lower(): put('enable_thinking', True, 'backend-template')
            elif 'gemma4' in model.lower(): put('enable_thinking', False, 'backend-template')
        if effective.get('reasoning_effort') == 'none': put('enable_thinking', False, 'launch')
    if settings and settings.get('effective'):
        # This integration starts the pinned strata-poly sampler without CLI
        # sampling overrides. src/poly/main.cpp::Sampling owns these defaults;
        # model-card defaults deliberately do not apply to the poly worker.
        if backend == 'polystrata':
            for key, value in {'temperature':0.0,'top_p':1.0,'top_k':20,'min_p':0.0,'seed':0}.items():
                if key not in effective: put(key, value, 'native-worker')
        for key, value in settings['effective'].items():
            if key in PARAMS: put(key, value)
        effort = effective.get('reasoning_effort')
        if effort is not None: put('enable_thinking', effort != 'none')
    if backend == 'polystrata' and effective.get('reasoning_effort') == 'high':
        effective['reasoning_effort'] = 'xhigh'
    capabilities = {'thinking': True, 'budget': True,
                    'preserve': backend in ('ninfer','polystrata','llamacpp'),
                    'efforts': ['low','medium','xhigh'] if backend == 'polystrata' else
                               ['minimal','low','medium','high','xhigh','max'],
                    'vision': bool(props.get('modalities', {}).get('vision', native.get('vision', False))),
                    'tools': False}
    if backend == 'radiance' and settings:
        caps=settings.get('capabilities',{})
        capabilities.update(thinking=bool(caps.get('reasoning')),budget=bool(caps.get('reasoning')),
            preserve=bool(props.get('chat_template_caps',{}).get('supports_preserve_reasoning')),
            efforts=caps.get('reasoning_efforts') or [],vision=bool(caps.get('image')))
    return {'schema_version':1, 'state':state['state'], 'instance':state.get('instance'),
            'backend':backend, 'model':props.get('model_name') or props.get('model_alias') or props.get('model'),
            'context':generation.get('n_ctx', launch.get('Context')),
            'limit':native.get('generation_limit') or generation.get('n_ctx') or launch.get('Context'),
            'effective':effective, 'sources':sources, 'capabilities':capabilities,
            'uses_template_effort': backend == 'radiance' and bool(settings) and 'reasoning_effort' in settings.get('request_defaults', {}) and settings['request_defaults']['reasoning_effort'] is None and 'reasoning_effort' not in settings['request_defaults'].get('chat_template_kwargs', {}),
            'budget_min':1 if backend=='ninfer' else -1 if backend=='radiance' else 0,
            'sampling_by_mode':native.get('sampling', {}),
            'sampling_by_effort':{('xhigh' if backend == 'polystrata' and k == 'high' else k): v
                                  for k, v in (settings or {}).get('sampling_by_effort', {}).items()},
            'history_export_url':None if backend=='radiance' else state.get('url','') + ('/kvmem-history-export' if backend=='polystrata' else '/history-export.html')}


class ChatBridge:
    def __init__(self, control):
        self.control = control
        self.lock = threading.RLock()
        self.active = {}
        self.records = {}

    def target(self, expected=None):
        state = self.control.status()
        if state['state'] != 'ready' or not state.get('url'):
            raise ValueError('No ready service; start a model first')
        if expected is not None and expected != state.get('instance'):
            raise ValueError('Service changed; refresh settings before sending')
        return state

    def get_json(self, state, path):
        parsed = urlsplit(state['url'])
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        try:
            headers = {}
            auth = self.control.monitor_authorization()
            if auth: headers['Authorization'] = auth
            conn.request('GET', path, headers=headers)
            response = conn.getresponse()
            body = response.read(4 * 1024 * 1024 + 1)
            if response.status != 200: raise ValueError('Engine settings unavailable')
            if len(body) > 4 * 1024 * 1024: raise ValueError('Engine settings too large')
            return json.loads(body)
        finally: conn.close()

    def context(self):
        state = self.control.status()
        if state['state'] != 'ready':
            return {'schema_version':1,'state':state['state'],'instance':state.get('instance'),
                    'effective':{},'capabilities':{},'backend':None,'model':None}
        props = self.get_json(state, '/props')
        backend=state['active_config']['parameters']['Backend']
        settings = self.get_json(state, '/settings' if backend=='polystrata' else '/server_info') if backend in ('polystrata','radiance') else None
        if state.get('instance') != self.control.status().get('instance'):
            raise ValueError('Service changed while reading settings')
        return context_payload(state, props, settings)

    def cancel(self, request_id):
        with self.lock:
            current = self.active.get(request_id)
            if current:
                current[1].set()
                conn = current[0]
                sock = current[2] if len(current)>2 else conn.sock
                if sock:
                    try: sock.shutdown(socket.SHUT_RDWR)
                    except OSError: pass
                conn.close()
            return {'cancelled':bool(current)}

    def cancel_all(self):
        with self.lock: ids = list(self.active)
        for key in ids: self.cancel(key)

    def record(self, request_id):
        with self.lock: record = copy.deepcopy(self.records.get(request_id))
        if record:
            snapshot, row, backend = {}, None, record.get('backend')
            try:
                state = self.target(record['instance'])
                backend = state.get('active_config', {}).get('parameters', {}).get('Backend')
                snapshot = self.get_json(state, '/api/monitor/snapshot')
                matches = [r for r in snapshot.get('active', []) + snapshot.get('recent', [])
                           if r.get('client_request_id') == request_id or
                           (record.get('native_id') and record['native_id'] in
                            (r.get('native_request_id'), r.get('id')))]
                if len(matches) == 1: row = matches[0]
            except (OSError, ValueError, http.client.HTTPException): pass
            metrics = request_metrics(record, snapshot, row, backend)
            if metrics: record['metrics'] = metrics
        return record

    def stream(self, handler, payload):
        request_id = handler.headers.get('X-KVMem-Request', '')
        if not handler.headers.get('X-KVMem-Instance'):
            raise ValueError('Service identity required; refresh settings before sending')
        if not request_id or len(request_id)>80 or not all(c.isalnum() or c in '-_' for c in request_id):
            raise ValueError('Invalid request identity')
        if not isinstance(payload, dict) or not isinstance(payload.get('messages'), list):
            raise ValueError('messages must be an array')
        payload = dict(payload, stream=True)
        # Reserve provider-managed tool execution for dedicated clients.
        if any(k in payload for k in ('strata_mcp','tools','tool_choice')):
            raise ValueError('This chat does not execute tools')
        raw = json.dumps(payload, ensure_ascii=False).encode()
        if len(raw)>MAX_BODY: raise ValueError('Chat request too large')
        with self.control.lock:
            state = self.target(handler.headers.get('X-KVMem-Instance'))
            if state.get('active_config', {}).get('parameters', {}).get('Backend') == 'polystrata':
                payload['_kvmem_request_id'] = request_id
                raw = json.dumps(payload, ensure_ascii=False).encode()
            elif state.get('active_config', {}).get('parameters', {}).get('Backend') == 'ninfer' and 'reasoning_budget_tokens' in payload:
                budget = payload.pop('reasoning_budget_tokens')
                if type(budget) is not int or budget < 1:
                    raise ValueError('NInfer thinking budget must be a positive integer')
                if 'thinking_budget' in payload and payload['thinking_budget'] != budget:
                    raise ValueError('Conflicting thinking budgets')
                payload['thinking_budget'] = budget
                raw = json.dumps(payload, ensure_ascii=False).encode()
            if state.get('active_config', {}).get('parameters', {}).get('Backend') == 'radiance':
                payload['stream_options']=dict(payload.get('stream_options') or {},include_usage=True)
                raw=json.dumps(payload,ensure_ascii=False).encode()
            parsed = urlsplit(state['url'])
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
            cancel = threading.Event()
            record = {'id':request_id,'instance':state.get('instance'),'state':'waiting',
                      'backend':state.get('active_config', {}).get('parameters', {}).get('Backend'),
                      'started_at_ms':int(time.time()*1000),'finish_reason':None,'timings':None}
            with self.lock:
                if request_id in self.records: raise ValueError('Duplicate request identity')
                self.active[request_id]=(conn,cancel)
                self.records[request_id]=record
                for key in list(self.records):
                    if len(self.records)<=100: break
                    if key not in self.active: self.records.pop(key)
            headers={'Content-Type':'application/json', 'Accept':'text/event-stream'}
            auth=self.control.monitor_authorization()
            if auth: headers['Authorization']=auth
            # Establish the connection before a stop/restart may reuse the port.
            try:
                conn.connect()
                conn.sock.settimeout(300)
                with self.lock: self.active[request_id]=(conn,cancel,conn.sock)
                conn.request('POST','/v1/chat/completions',raw,headers)
            except Exception:
                with self.lock:
                    self.active.pop(request_id,None)
                    record['state']='error'
                    record['finished_at_ms']=int(time.time()*1000)
                conn.close()
                raise
        chunks=queue.Queue(maxsize=64)
        def enqueue(value):
            while not cancel.is_set():
                try: chunks.put(value,timeout=.2);return
                except queue.Full: pass
        def read():
            try:
                response=conn.getresponse()
                if response.status!=200:
                    detail=response.read(16384).decode('utf-8',errors='replace')
                    enqueue(('error',f'Engine HTTP {response.status}: {detail}'))
                else:
                    for line in iter(response.readline,b''):
                        if cancel.is_set(): break
                        enqueue(('line',line))
                    enqueue(('end',None))
            except Exception as error:
                enqueue(('error',str(error)))
        reader=threading.Thread(target=read,daemon=True);reader.start()
        done=False;last_beat=time.monotonic()
        try:
            handler.close_connection=True
            handler.send_response(200)
            handler.send_header('Content-Type','text/event-stream; charset=utf-8')
            handler.send_header('Cache-Control','no-store')
            handler.send_header('X-Accel-Buffering','no')
            handler.send_header('Connection','close')
            handler.end_headers()
            while not cancel.is_set():
                # Detect a closed browser even when prefill produces no SSE data.
                if select.select([handler.connection],[],[],0)[0]:
                    if not handler.connection.recv(1,socket.MSG_PEEK): break
                try: kind,data=chunks.get(timeout=.2)
                except queue.Empty:
                    if time.monotonic()-last_beat>5:
                        handler.wfile.write(b': waiting\n\n');handler.wfile.flush();last_beat=time.monotonic()
                    continue
                if kind=='error': raise RuntimeError(data)
                if kind=='end':
                    if not done: raise RuntimeError('Engine stream ended before DONE')
                    break
                if data.startswith(b'data:'):
                    value=data[5:].strip()
                    if value==b'[DONE]': done=True
                    else:
                        event=json.loads(value)
                        if event.get('error'): raise RuntimeError(str(event['error']))
                        with self.lock:
                            record['native_id']=event.get('id',record.get('native_id'))
                            for choice in event.get('choices',[]):
                                delta=choice.get('delta',{})
                                if delta.get('reasoning_content') or delta.get('content'):
                                    record.setdefault('first_token_at_ms',int(time.time()*1000))
                                    record['state']='thinking' if delta.get('reasoning_content') else 'answering'
                                if choice.get('finish_reason'): record['finish_reason']=choice['finish_reason']
                            if event.get('timings'): record['timings']=event['timings']
                            if event.get('usage'): record['usage']=event['usage']
                        if record['backend']=='radiance':
                            speculative_event(event)
                            data=b'data: '+json.dumps(event,ensure_ascii=False).encode()+b'\n'
                handler.wfile.write(data);handler.wfile.flush()
                if done:
                    handler.wfile.write(b'\n');handler.wfile.flush()
                    break
        except (BrokenPipeError,ConnectionResetError,OSError):
            pass
        except Exception as error:
            with self.lock: record['state']='error';record['finish_reason']='error'
            try:
                handler.wfile.write(b'data: '+json.dumps({'error':{'message':str(error)}}).encode()+b'\n\ndata: [DONE]\n\n')
                handler.wfile.flush()
            except OSError: pass
        finally:
            self.cancel(request_id)
            reader.join(timeout=1)
            with self.lock:
                self.active.pop(request_id,None)
                if record['state']!='error': record['state']='completed' if done else 'cancelled'
                record['finished_at_ms']=int(time.time()*1000)
