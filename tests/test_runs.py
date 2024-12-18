import os
import unittest

from stockml.runs import from_file, get_path_in

class TestDatasourceMethods(unittest.TestCase):

    def test_run_name_inference(self):
        """
        Tests that run names can be inferred from incomplete run names.
        Run names will match the first run which starts with the partial run name.
        """
        # Try to get run with run name set to 'tes' (part of test_run).
        run_data, run_name = from_file("tests/runs/test_misc.json", "tes")
        
        # Test that the run name is as expected.
        self.assertEqual(run_name, "test_run")

    def test_env_var_and_global_run(self):
        """Tests that environment variables are copied into run_data correctly,
        and that global values get copied into runs correctly.
        """
        os.environ['API_KEY'] = "test"
        run_data, run_name = from_file("tests/runs/test_env.json", "test_run")
        
        # Test that the api key was retrieved from the environment variable correctly
        api_key = get_path_in("dataset.api_key", run_data)
        self.assertEqual(api_key, "test")
    
    def test_source_wildcard(self):
        run_data, run_name = from_file("tests/runs/test_source_wildcard.json", "test_run")
    
        # Make sure the wildcard itself isn't included in the run
        self.assertFalse(get_path_in("dataset.sources.*", run_data))
    
        # Make sure the test key was included in both sources
        test_key = { "a": 123 }
        
        test_key_a = get_path_in("dataset.sources.source_a.test_key", run_data)
        self.assertEqual(test_key_a, test_key)
        
        test_key_b = get_path_in("dataset.sources.source_b.test_key", run_data)
        self.assertEqual(test_key_b, test_key)
        

if __name__ == '__main__':
    unittest.main()