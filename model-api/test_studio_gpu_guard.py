import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from model_switch_api import Backend, LLAMA, native_studio_models

class GuardTests(unittest.TestCase):
    def test_only_live_studio_llama_process_matches(self):
        with tempfile.TemporaryDirectory() as root:
            for pid, group, exe, state in [
                (1,'shiba-unsloth-studio.service',LLAMA,'S'),
                (2,'shiba-model-api.service',LLAMA,'S'),
                (3,'shiba-unsloth-studio.service','/usr/bin/python3','S'),
                (4,'shiba-unsloth-studio.service',LLAMA,'Z')]:
                p=Path(root)/str(pid);p.mkdir()
                (p/'cgroup').write_text(group);(p/'cmdline').write_bytes(exe.encode()+b'\0')
                (p/'status').write_text('State:\t'+state+'\n')
            self.assertEqual(native_studio_models(Path(root)),[1])
    def test_guard_prevents_launch_without_stopping_anything(self):
        backend=Backend({'test':{'available':True}})
        with patch('model_switch_api.native_studio_models',return_value=[1]), patch.object(backend,'stop') as stop:
            with self.assertRaisesRegex(RuntimeError,'occupied'):
                backend.ensure('test')
            stop.assert_not_called()

if __name__=='__main__': unittest.main()
