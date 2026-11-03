import json
import tempfile
import unittest
from pathlib import Path

from matrix_db import init_db
from matrix_reanalyze import final_attempt, reanalyze


class ReanalysisTests(unittest.TestCase):
    def test_final_attempt_isolated_and_checked(self):
        text = ("# matrix job 00001 seq=1 run=r universe=u\nold\n"
                "# matrix job 00001 seq=1 run=r universe=u\nnew\n")
        attempt, count = final_attempt(text, "r", 1, 1, "u")
        self.assertEqual(count, 2)
        self.assertNotIn("old", attempt)
        self.assertIn("new", attempt)
        with self.assertRaises(ValueError):
            final_attempt(text, "r", 2, 1, "u")
        with self.assertRaises(ValueError):
            final_attempt("no header", "r", 1, 1, "u")

    def test_readonly_reanalysis_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "ledger.sqlite3"
            conn = init_db(db)
            conn.execute("INSERT INTO runs (run_id, universe_hash) VALUES ('r', 'u')")
            log = root / "job.log"
            header = "# matrix job 00001 seq=1 run=r universe=u\n"
            log.write_text(header + "Gathered output[0] from all ranks:\n[[1]]\n"
                           "Gathered output sums from all ranks:\n[[1]]\n" + header +
                           "Gathered output[0] from all ranks:\n[[2]]\n"
                           "Gathered output sums from all ranks:\n[[2]]\n")
            conn.execute("INSERT INTO jobs (run_id,job_id,seq,status,attempts,log_path,model,steps,clients,batch_size) VALUES (?,?,?,?,?,?,?,?,?,?)",
                         ("r", 1, 1, "success", 2, str(log), "perfect", 1, 1, 1))
            conn.execute("INSERT INTO jobs (run_id,job_id,seq,status,attempts,log_path) VALUES (?,?,?,?,?,?)",
                         ("r", 2, 2, "success", 1, str(root / "missing")))
            conn.execute("INSERT INTO jobs (run_id,job_id,seq,status) VALUES (?,?,?,?)", ("r", 3, 3, "failed"))
            conn.commit()
            conn.close()
            original = db.read_bytes()
            summary = reanalyze(db, "r", root / "output")
            self.assertEqual(db.read_bytes(), original)
            self.assertEqual(summary["counts"], {"jobs": 3, "successful_jobs": 2,
                             "parsed": 1, "errors": 1, "groups": 1, "anomalies": 0})
            self.assertEqual(summary["jobs"][0]["result"]["first"], [[2]])
            self.assertIn("FileNotFoundError", summary["jobs"][1]["error"])
            self.assertEqual(json.loads((root / "output" / "provenance.json").read_text()), summary)
            with self.assertRaises(FileExistsError):
                reanalyze(db, "r", root / "output")
            self.assertEqual(db.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
