"""The GH200 hosts' shared GPU lease (flock on ~/.cache/vesuvius-gpu/exclusive.lock + donor-priority.json owner record).

Optional: with STUDIO_GPU_LEASE=0 (or no lease directory) nothing is taken, so the code runs on any machine.
`acquire()` waits for the lease instead of preempting its holder; manual-revision rounds on the host may ask the holder
to release with SIGTERM, which train_cpt.py answers with a checkpoint and a clean exit (run_cpt.sh then re-acquires).
"""
import contextlib, fcntl, json, os, time, uuid
from pathlib import Path

ROOT = Path(os.environ.get('VESUVIUS_GPU_LEASE_DIR', str(Path.home() / '.cache/vesuvius-gpu')))


@contextlib.contextmanager
def acquire(requester, reason, wait=True, poll=20):
    if os.environ.get('STUDIO_GPU_LEASE', '1') == '0' or not ROOT.exists():
        yield None; return
    gate = ROOT / 'donor-priority.json'
    with (ROOT / 'exclusive.lock').open('a') as lock:
        announced = False
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                previous = json.loads(gate.read_text()) if gate.exists() else {'status': 'released'}
                if previous.get('status') in ('released', 'complete', 'cancelled') or not _alive(previous.get('pid')):
                    break
                fcntl.flock(lock, fcntl.LOCK_UN)
            except BlockingIOError:
                pass
            if not wait:
                raise RuntimeError('GPU lease is occupied')
            if not announced:
                print('WAITING_FOR_SHARED_GPU', flush=True); announced = True
            time.sleep(poll)
        token = str(uuid.uuid4())
        mine = dict(status='running', requester=requester, request_id=token, ownership_token=token, pid=os.getpid(), reason=reason)

        def publish(value):
            tmp = gate.with_name(gate.name + '.' + token + '.tmp'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(gate)
        publish(mine); print('GPU_LEASE_ACQUIRED', token, flush=True)
        code = 1
        try:
            yield token
            code = 0
        finally:
            try:
                if json.loads(gate.read_text()).get('ownership_token') == token:
                    publish(dict(mine, status='released', exit_code=code))
            except Exception:
                pass


def _alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0); return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
