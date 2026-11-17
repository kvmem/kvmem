"""Model-free HTTP fixture; only worker execution/inspection are simulated."""
import json, secrets, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
from test_console_memory import console, evidence
from test_auto_budget import budget_evidence
console.AUTO_BUDGET_SUPPORTED = True

control = console.Controller(console.ROOT, sys.argv[2], int(sys.argv[1]))
original_status = control.status
verified = False

def status():
    value = original_status()
    if verified:
        value.update(state='ready', url='http://127.0.0.1:18470')
        if value.get('memory_trial'): value['memory_trial']['startup_verified'] = True
    return value

def start(data):
    global verified
    control.active = console.normalize(data, control.backends)
    p = control.active['parameters']
    control.instance = secrets.token_hex(12)
    control.memory_diagnosis_instance = None
    control.memory_trial = None
    control.launch_was_ready = False
    verified = p['Budget'] != 65536
    control.exit_code = None if verified else 1
    _, tokens, log = evidence(parameters=p)
    control.active_plan = {'token_settings':tokens}
    control.log_path.write_text(log)
    return status()

original_stop = control.stop
def stop():
    global verified
    verified=False
    return original_stop()

control.status, control.start, control.stop = status, start, stop
control.inspect_model = lambda *a,**kw: {'schema_version':1,'status':'pass','can_start':True,'model':{},'features':{},'checks':[]}
def budget_launch(data):
    p=console.normalize(data,control.backends)['parameters']
    tokens,report=budget_evidence(dict(console.DEFAULTS,**p))
    return {'argv':['fixture-worker'],'worker':{'budget_planning_version':1},'token_settings':tokens,'report':report}
def budget_worker(plan):
    time.sleep(.3)
    return 'KVMEM_BUDGET_PLAN '+json.dumps(plan['report'])
control.plan,control._run_budget_plan=budget_launch,budget_worker
server = console.Server(('127.0.0.1',int(sys.argv[1])),console.Handler)
server.control=control
server.gpus=[{'uuid':'GPU-12345678','name':'Test GPU','memory_mib':'8192'}]
server.hardware=console.SystemHardwareSampler()
server.serve_forever()
