import unittest
from unittest.mock import patch
from app.analyzers.bug_investigator import investigate_bug


class TestBugInvestigator(unittest.TestCase):

    @patch("app.analyzers.bug_investigator.ask_llm")
    def test_investigate_bug_structure(self, mock_ask):
        mock_ask.return_value = "## Root Cause Diagnosis\nPotential NoneType error in scan."

        result = investigate_bug(
            bug_description="scan_repository fails when scanning empty directory",
            error_trace="ValueError: Invalid repository directory"
        )

        self.assertIn("bug_description", result)
        self.assertIn("diagnosis", result)
        self.assertIn("affected_files", result)
        self.assertIn("sources", result)
        self.assertEqual(result["bug_description"], "scan_repository fails when scanning empty directory")
        self.assertIn("Root Cause", result["diagnosis"])


if __name__ == "__main__":
    unittest.main()

