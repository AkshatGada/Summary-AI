# summarizer/test_summarizer.py

import unittest
from summarizer import Summarizer

class TestSummarizer(unittest.TestCase):
    def setUp(self):
        self.summarizer = Summarizer()

    def test_empty_input(self):
        summary = self.summarizer.summarize("")
        self.assertIsNone(summary, "Empty input should return None.")

    def test_standard_summarization(self):
        text = (
            "Artificial intelligence (AI) is intelligence demonstrated by machines, "
            "in contrast to the natural intelligence displayed by humans and animals. "
            "Leading AI textbooks define the field as the study of 'intelligent agents': "
            "any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals."
        )
        summary = self.summarizer.summarize(text)
        self.assertIsInstance(summary, str, "The summary should be a string.")
        self.assertTrue(len(summary) > 0, "The summary should not be empty.")

    def test_long_text_summarization(self):
        # Generate a long text by repeating a paragraph.
        paragraph = (
            "In the rapidly evolving world of technology, companies are continually seeking innovative ways to stay ahead. "
            "This involves adopting cutting-edge techniques and constantly refining existing processes. "
            "As a result, there is a growing emphasis on research and development across various sectors. "
        )
        long_text = paragraph * 20  # create a long text by repeating the paragraph
        final_summary = self.summarizer.summarize_long_text(long_text)
        self.assertIsInstance(final_summary, str, "The final summary should be a string.")
        self.assertTrue(len(final_summary) > 0, "The final summary should not be empty.")

if __name__ == '__main__':
    unittest.main()
