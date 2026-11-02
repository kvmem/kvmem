"""Model-specific tokenizer cache publication and identity checks."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from assets import FILES, tokenizer_path


class AssetsTest(unittest.TestCase):
    def test_gemma_json_merges_are_published_and_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';(source/'tools').mkdir(parents=True)
            for name in ('strata_tokenizer.py','gguf_reader.py'):(source/'tools'/name).write_text('exporter')
            model=root/'model.gguf';model.write_bytes(b'GGUF fixture')
            def extract(model,out):
                target=out/'tokenizer';target.mkdir()
                for name in FILES:
                    (target/('merges.json' if name=='merges.txt' else name)).write_text('valid '+name)
            with patch.dict('sys.modules',{'strata_tokenizer':SimpleNamespace(extract=extract)}):
                target=tokenizer_path(model,source,prepare=True,cache=root/'cache')
                self.assertEqual(tokenizer_path(model,source,prepare=True,cache=root/'cache'),target)
                (target/'merges.json').write_text('modified newline merges')
                with self.assertRaisesRegex(ValueError,'modified'):
                    tokenizer_path(model,source,prepare=True,cache=root/'cache')

    def test_identity_and_complete_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';(source/'tools').mkdir(parents=True)
            for name in ('strata_tokenizer.py','gguf_reader.py'):
                (source/'tools'/name).write_text('exporter v1')
            model=root/'model.gguf';model.write_bytes(b'GGUF fixture')
            cache=root/'cache';calls=[]
            def extract(model, out):
                calls.append(model);target=out/'tokenizer';target.mkdir()
                for name in FILES:(target/name).write_text('valid '+name)
            with patch.dict('sys.modules', {'strata_tokenizer':SimpleNamespace(extract=extract)}):
                planned=tokenizer_path(model,source,prepare=False,cache=cache)
                self.assertFalse(cache.exists())
                actual=tokenizer_path(model,source,prepare=True,cache=cache)
                self.assertEqual(actual,planned)
                self.assertEqual(tokenizer_path(model,source,prepare=True,cache=cache),actual)
                self.assertEqual(len(calls),1)
                model.write_bytes(b'GGUF changed fixture')
                other=tokenizer_path(model,source,prepare=True,cache=cache)
                self.assertNotEqual(other,actual)
                (other/'chat_template.jinja').write_text('modified')
                with self.assertRaisesRegex(ValueError,'modified'):
                    tokenizer_path(model,source,prepare=True,cache=cache)

    def test_missing_template_never_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'tools').mkdir()
            for name in ('strata_tokenizer.py','gguf_reader.py'):(root/'tools'/name).touch()
            model=root/'model.gguf';model.write_bytes(b'GGUF fixture')
            cache=root/'cache'
            def extract(model,out):
                (out/'tokenizer').mkdir()
                for name in FILES[:-1]:(out/'tokenizer'/name).touch()
            with patch.dict('sys.modules', {'strata_tokenizer':SimpleNamespace(extract=extract)}):
                with self.assertRaisesRegex(ValueError,'complete tokenizer'):
                    tokenizer_path(model,root,prepare=True,cache=cache)
            self.assertEqual(list(cache.iterdir()),[])


if __name__=='__main__':unittest.main()
