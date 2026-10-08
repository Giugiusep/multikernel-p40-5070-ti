import unittest
from hf_gpt_backend import render_prompt, request_options

class Contracts(unittest.TestCase):
    def test_native_limit_includes_output(self):
        for context in [512, 1024]:
            self.assertEqual(request_options({'max_tokens': 32}, context-32, context)['max_new_tokens'], 32)
            with self.assertRaises(ValueError):
                request_options({'max_tokens': 33}, context-32, context)

    def test_greedy_does_not_set_sampling_parameters(self):
        opts=request_options({'temperature':0,'max_tokens':16},4,512)
        self.assertFalse(opts['do_sample'])
        self.assertNotIn('temperature',opts)

    def test_plain_chat_mapping_preserves_roles(self):
        self.assertEqual(render_prompt({'messages':[{'role':'user','content':'Hello'}]},True), 'User: Hello\nAssistant:')
        with self.assertRaises(ValueError):
            render_prompt({'messages':[{'role':'user','content':[{'type':'image_url'}]}]},True)

    def test_invalid_and_unsupported_requests(self):
        for payload in [{'max_tokens':0},{'temperature':float('nan')},{'top_p':0},{'presence_penalty':1}]:
            with self.assertRaises(ValueError):request_options(payload,3,512)
        with self.assertRaises(ValueError):render_prompt({'prompt':'Hi','tools':[{}]},False)

if __name__=='__main__':unittest.main()
