"""Check native UI retirement without loading a model or GPU."""
import http.client
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys,threading,unittest
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'backends/polystrata/terminal'))
from serve.server import make_handler

class Routes(unittest.TestCase):
    def test_only_export_remains_and_api_stays_available(self):
        svc=SimpleNamespace(kvmem_console_only=True,cors_origins=[],api_key='',host_names={'127.0.0.1','localhost'},allowed_hosts=[],settings_payload=lambda:{'effective':{'max_tokens':128}})
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(svc))
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            for path in ['/','/web/app.js','/api-monitor','/fonts/outfit.woff2']:
                c=http.client.HTTPConnection('127.0.0.1',server.server_port);c.request('GET',path);r=c.getresponse();self.assertEqual(r.status,404);r.read();c.close()
            c=http.client.HTTPConnection('127.0.0.1',server.server_port);c.request('GET','/kvmem-history-export');r=c.getresponse();self.assertEqual(r.status,200);self.assertIn(b'indexedDB.databases',r.read());c.close()
            c=http.client.HTTPConnection('127.0.0.1',server.server_port);c.request('GET','/settings');r=c.getresponse();self.assertEqual(r.status,200);self.assertIn(b'max_tokens',r.read());c.close()
        finally:server.shutdown();server.server_close()
if __name__=='__main__':unittest.main()
