import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.utils.summarize_results_csv import ALIASES, COLUMNS, summarize


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'results.csv'

    def write(self, rows, aliases=False):
        with self.path.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            writer.writerow([ALIASES[c] for c in COLUMNS] if aliases else COLUMNS)
            writer.writerows(rows)

    def test_zero_steering_counts_in_means_but_not_episode_ratio(self):
        for aliases in (False, True):
            self.write([[100, 0, 0, 0, 0], [50, 4, 3, 1, 2], [0, 1, 1, 0, 1]], aliases)
            result = summarize(self.path)
            self.assertEqual(result['episodes'], 3)
            self.assertEqual(result['average_progress_percent'], 50)
            self.assertEqual(result['average_steering_times'], 5 / 3)
            self.assertEqual(result['average_prompt_alignment'], 60)
            self.assertEqual(summarize(self.path, 'episode')['average_prompt_alignment'], 75)
            self.assertEqual(summarize(self.path, 'count')['average_prompt_alignment'], 1)

    def test_all_zero_steering_has_no_ratio(self):
        self.write([[100, 0, 0, 0, 0]])
        result = summarize(self.path)
        self.assertIsNone(result['average_prompt_alignment'])
        json.dumps(result, allow_nan=False)

    def test_invalid_rows_and_method(self):
        for row in [[101, 0, 0, 0, 0], [50, 1, 0, 0, 2], [50, 1.5, 0, 0, 1], ['nan', 0, 0, 0, 0]]:
            self.write([row])
            with self.assertRaises(ValueError):
                summarize(self.path)
        with self.assertRaises(ValueError):
            summarize(self.path, 'unknown')
        self.write([])
        with self.assertRaisesRegex(ValueError, 'no episode'):
            summarize(self.path)

    def test_cli_wont_overwrite_input(self):
        self.write([[100, 0, 0, 0, 0]])
        original = self.path.read_bytes()
        result = subprocess.run([sys.executable, '-m', 'scripts.utils.summarize_results_csv',
                                 str(self.path), '-o', str(self.path)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
