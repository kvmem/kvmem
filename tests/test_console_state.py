"""Private state creation preserves existing access and is safe to repeat."""
import os
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from console_state import prepare_windows_state_directory

@unittest.skipUnless(os.name=='nt','Windows ACL contract')
class StateDirectoryTest(unittest.TestCase):
    def test_create_reopen_and_preserve_existing_contents(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)/'parent'/'private'
            prepare_windows_state_directory(directory)
            log=directory/'service.log'
            log.write_text('saved')
            prepare_windows_state_directory(directory)
            self.assertEqual(log.read_text(),'saved')
            log.write_text('reopened')
            self.assertEqual(log.read_text(),'reopened')

if __name__=='__main__':unittest.main()
