import unittest
from unittest.mock import patch, Mock
import json
import os
from scripts.bigenrollment import query_llm

api_key = os.getenv("TOGETHER_API_KEY_dafhe")

class TestQueryLLM(unittest.TestCase):
    def setUp(self):
        self.valid_response = {
            "gender": {"F": 0.3, "M": 0.7},
            "religion": {
                "Christian": 0.35,
                "Hindu": 0.25,
                "Muslim": 0.15,
                "None": 0.10,
                "Other": 0.15
            }
        }
        
        # Mock response object structure
        self.mock_response = Mock()
        self.mock_response.choices = [Mock()]
        self.mock_response.choices[0].message = Mock()
        self.mock_response.choices[0].message.content = json.dumps(self.valid_response)

    @patch.dict(os.environ, {'TOGETHER_API_KEY_dafhe': 'api-key'})
    @patch('together.Together')
    def test_successful_query(self, mock_together):
        # Setup mock
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = self.mock_response
        mock_together.return_value = mock_client

        # Test
        result = query_llm(2020)

        # Verify
        self.assertEqual(result["gender"]["F"], 0.6)
        self.assertEqual(result["gender"]["M"], 0.4)
        self.assertEqual(result["religion"]["Christian"], 0.35)
        self.assertEqual(sum(result["gender"].values()), 1.0)
        self.assertEqual(sum(result["religion"].values()), 1.0)

    def test_missing_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            result = query_llm(2020)
            
            # Should return default distributions
            self.assertEqual(result["gender"], {"F": 0.6, "M": 0.4})
            self.assertEqual(result["religion"]["Christian"], 0.35)
            self.assertEqual(sum(result["gender"].values()), 1.0)
            self.assertEqual(sum(result["religion"].values()), 1.0)

    @patch.dict(os.environ, {'TOGETHER_API_KEY_dafhe': 'fake-key'})
    @patch('together.Together')
    def test_api_error_handling(self, mock_together):
        # Setup mock to raise exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_together.return_value = mock_client

        # Should return default values on error
        result = query_llm(2020)
        self.assertEqual(result["gender"], {"F": 0.6, "M": 0.4})

    @patch.dict(os.environ, {'TOGETHER_API_KEY_dafhe': 'fake-key'})
    @patch('together.Together')
    def test_invalid_json_response(self, mock_together):
        # Setup mock with invalid JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Invalid JSON {["
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_together.return_value = mock_client

        # Should return default values on invalid JSON
        result = query_llm(2020)
        self.assertEqual(result["gender"], {"F": 0.6, "M": 0.4})

    @patch.dict(os.environ, {'TOGETHER_API_KEY_dafhe': 'fake-key'})
    @patch('together.Together')
    def test_invalid_distribution_values(self, mock_together):
        # Setup mock with invalid distribution (doesn't sum to 1.0)
        invalid_response = {
            "gender": {"F": 0.3, "M": 0.3},  # Sums to 0.6
            "religion": {
                "Christian": 0.35,
                "Hindu": 0.25,
                "Muslim": 0.15,
                "None": 0.10,
                "Other": 0.15
            }
        }
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = json.dumps(invalid_response)
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_together.return_value = mock_client

        # Should return default values when validation fails
        result = query_llm(2020)
        self.assertEqual(result["gender"], {"F": 0.6, "M": 0.4})

if __name__ == '__main__':
    unittest.main()