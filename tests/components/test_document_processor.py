import unittest
from unittest.mock import MagicMock
from components.document_processor import DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        """Set up mock ChromaDB client and collection for testing."""
        self.mock_client = MagicMock()
        self.mock_collection = MagicMock()
        self.mock_client.get_or_create_collection.return_value = self.mock_collection
        self.processor = DocumentProcessor(chroma_client=self.mock_client, collection_name="multilingual_documents")

    def test_initialization(self):
        """Test that we are passing the collection name correctly during initialization."""
        self.mock_client.get_or_create_collection.assert_called_with("multilingual_documents")
        self.assertEqual(self.processor.collection, self.mock_collection)

    def test_get_collection(self):
        """Test that get_collection returns the correct collection."""
        collection = self.processor.get_collection()
        print(f"Collection returned: {collection}")
        self.assertEqual(collection, self.mock_collection)

if __name__ == "__main__":
    unittest.main()