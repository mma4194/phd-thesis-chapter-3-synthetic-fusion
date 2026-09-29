import ast, contextlib, io, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import worker

class SourceContextTests(unittest.TestCase):
    def test_real_guard_reads_current_source_and_keeps_duplicate_check(self):
        source=(ROOT/'source_cells/residential/053.py').read_text()
        tree=ast.parse(source)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_is_counter_progression_publication_guard_col_12c2')
        block=next(n for n in tree.body if isinstance(n,ast.Try) and '_counter_guard_definition_count_12c2' in ast.get_source_segment(source,n))
        # Use the actual source guard and function declaration, without fitting models.
        code=ast.get_source_segment(source,fn)+'\n'+ast.get_source_segment(source,block)+'\nobserved_file=__file__\nassert _counter_guard_definition_count_12c2 == 1\n'
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'source_cells/probe';folder.mkdir(parents=True)
            p=folder/'001.py';p.write_text(code);run=root/'run';run.mkdir()
            ns={'__file__':'master_notebook.ipynb'}
            with patch.object(worker,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):
                worker.exec_cell('probe',1,ns,run)
                self.assertEqual(ns['_counter_guard_definition_count_12c2'],1)
                self.assertEqual(ns['observed_file'],str(p))
                self.assertEqual(ns['__file__'],'master_notebook.ipynb')
                p.write_text(ast.get_source_segment(source,fn)+'\n'+code)
                with self.assertRaises(AssertionError):worker.exec_cell('probe',1,ns,run)
                self.assertEqual(ns['_counter_guard_definition_count_12c2'],2)
                self.assertEqual(ns['__file__'],'master_notebook.ipynb')
    def test_absent_file_context_is_restored_after_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'source_cells/probe';folder.mkdir(parents=True)
            (folder/'001.py').write_text('raise ValueError("intentional failure")')
            ns={}
            with patch.object(worker,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):worker.exec_cell('probe',1,ns,root)
            self.assertNotIn('__file__',ns)
    def test_distributed_guard_has_one_definition(self):
        source=(ROOT/'source_cells/residential/053.py').read_text()
        self.assertEqual(source.count('def '+'_is_counter_progression_publication_guard_col_12c2'),1)

if __name__=='__main__':unittest.main()
