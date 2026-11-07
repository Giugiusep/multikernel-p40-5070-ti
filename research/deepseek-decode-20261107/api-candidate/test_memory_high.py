import unittest
from model_switch_api import Backend
class MemoryHighTests(unittest.TestCase):
 def entry(self,**changes):
  e={'runner':'strata','path':'/tmp/ds4.json','memory_limit_gib':42};e.update(changes);return e
 def test_default_soft_limit_preserved(self):
  command,_=Backend({}).command(self.entry());self.assertIn('--property=MemoryHigh=38G',command);self.assertIn('--property=MemoryMax=42G',command)
 def test_explicit_soft_limit_keeps_hard_and_swap_limits(self):
  command,_=Backend({}).command(self.entry(memory_high_gib=40));self.assertIn('--property=MemoryHigh=40G',command);self.assertIn('--property=MemoryMax=42G',command);self.assertIn('--property=MemorySwapMax=0',command)
 def test_reject_invalid_soft_limits(self):
  for value in [0,-1,42,43]:
   with self.subTest(value=value),self.assertRaises(ValueError):Backend({}).command(self.entry(memory_high_gib=value))
 def test_require_hard_limit(self):
  with self.assertRaises(ValueError):Backend({}).command({'runner':'strata','path':'/tmp/ds4.json','memory_high_gib':40})
 def test_null_soft_limit_uses_existing_default(self):
  command,_=Backend({}).command(self.entry(memory_high_gib=None));self.assertIn('--property=MemoryHigh=38G',command)
if __name__=='__main__':unittest.main()
