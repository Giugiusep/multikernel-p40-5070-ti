import unittest
from model_switch_api import Backend, STRATA_SERVER, BACKEND_PORT

class LaunchTests(unittest.TestCase):
    def test_existing_qwen_launch_unchanged(self):
        entry={'runner':'strata','path':'/tmp/qwen.json','python':'/tmp/python'}
        cmd,env=Backend({}).command(entry)
        self.assertEqual(cmd,['/tmp/python',STRATA_SERVER,'--engine','strata','--config','/tmp/qwen.json','--host','127.0.0.1','--port',str(BACKEND_PORT)])
        self.assertEqual(env['STRATA_DIAG_BYPASS_EXPERT_HITS'],'1')
    def test_deepseek_is_isolated_and_memory_bounded(self):
        entry={'runner':'strata','path':'/tmp/deepseek.json','python':'/tmp/python','server':'/tmp/new-server.py','memory_limit_gib':42,'diagnostic_no_gpu_experts':False}
        cmd,env=Backend({}).command(entry)
        self.assertEqual(cmd[:4],['systemd-run','--user','--scope','--quiet'])
        self.assertIn('--property=MemoryMax=42G',cmd)
        self.assertIn('--property=MemorySwapMax=0',cmd)
        self.assertIn('/tmp/new-server.py',cmd)
        self.assertNotIn('STRATA_DIAG_BYPASS_EXPERT_HITS',env)

if __name__=='__main__':unittest.main()
