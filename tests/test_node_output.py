"""The node's caption == a direct capgen call on the runtime dir (same tags, seed, options).

    KREA2_CAPGEN_RUNTIME=<capgen runtime dir of the vendored version> python -m unittest discover -s tests
"""
import importlib.util
import json
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RT = os.environ.get('KREA2_CAPGEN_RUNTIME')
CASES = [('cat ears, school uniform, -white background', 'long', 7, 1.0, False, True, True),   # caption()'s order
         ('blue archive\nbeach, -simple background', 'short', 9007199254740991, 0.3, True, False, False)]
DIRECT = '''import json, sys
sys.path.append(sys.argv[1])
import capgen
g = capgen.Generator.load(sys.argv[1] + '/model')
print(json.dumps([g.generate(t.replace('\\n', ','), n=1, length=l, seed=s, wild=w, overlay_text=o, scene_text=sc,
                             mix_look=m)[0].text for t, l, s, w, o, sc, m in json.loads(sys.argv[2])]))
'''


def load_bridge():
    spec = importlib.util.spec_from_file_location('capgen_bridge', os.path.join(ROOT, 'capgen_bridge.py'))
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    return bridge


@unittest.skipUnless(RT, 'set KREA2_CAPGEN_RUNTIME to a capgen runtime dir')
class NodeOutput(unittest.TestCase):
    def test_same_caption_as_capgen(self):
        bridge = load_bridge()
        got = [bridge.caption(*c)[0] for c in CASES]
        want = json.loads(subprocess.run([sys.executable, '-I', '-B', '-c', DIRECT, RT, json.dumps(CASES)], check=True,
                                         capture_output=True, text=True, cwd=RT).stdout)
        self.assertEqual(got, want)
        self.assertTrue(all(got))
        self.assertFalse({'capgen', 'tagdict', 'textrules'} & set(sys.modules))    # private package names only


class PinPaths(unittest.TestCase):
    def test_unsafe_paths_refused(self):
        bridge = load_bridge()
        for rel in ('../x', '/etc/passwd', 'model/../../x', 'a\\b', 'c:/x', 'model//x', './x', ''):
            self.assertRaises(ValueError, bridge.check_rel, rel)
        bridge.check_rel('model/arrays.npz')

    def test_unrewritable_vendored_import_aborts(self):
        spec = importlib.util.spec_from_file_location('vendor_tool', os.path.join(ROOT, 'tools', 'vendor_runtime.py'))
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        for src in ('if True: import tagdict as TD', 'import os, textrules', 'x = 1; from capgen import G'):
            self.assertRaises(SystemExit, tool.rewrite_imports, 'm.py', src, ['capgen', 'tagdict', 'textrules'])


if __name__ == '__main__':
    unittest.main()
