# summarizer/summarizer.py

import logging
from typing import List, Optional
from transformers import pipeline

logging.basicConfig(level=logging.INFO)


class Summarizer:
    """
    A class to perform text summarization using HuggingFace transformers.
    """

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the summarizer with the specified model.
        
        :param model_name: The HuggingFace model name to use for summarization.
        """
        self.model_name = model_name
        try:
            self.summarizer = pipeline("summarization", model=self.model_name)
            logging.info(f"Loaded summarization model: {self.model_name}")
        except Exception as e:
            logging.error(f"Error loading summarization model {self.model_name}: {e}")
            raise

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40, do_sample: bool = False) -> Optional[str]:
        """
        Summarize the provided text using the summarization pipeline.
        
        :param text: The text to summarize.
        :param max_length: Maximum length of the summary.
        :param min_length: Minimum length of the summary.
        :param do_sample: Whether to use sampling; if False, the summary is deterministic.
        :return: A summarized string or None if summarization fails.
        """
        if not text:
            logging.warning("Empty text provided for summarization.")
            return None

        try:
            summary_output = self.summarizer(
                text, 
                max_length=max_length, 
                min_length=min_length, 
                do_sample=do_sample
            )
            if summary_output and isinstance(summary_output, list):
                summary = summary_output[0].get('summary_text', None)
                logging.debug(f"Summarization successful. Summary: {summary}")
                return summary.strip() if summary else None
            else:
                logging.error("Unexpected output format from summarization pipeline.")
                return None
        except Exception as e:
            logging.error(f"Error during summarization: {e}")
            return None

    def summarize_long_text(
        self,
        text: str,
        chunk_size: int = 300,  # number of words per chunk (adjust based on model limits)
        overlap: int = 50,      # overlapping words between chunks to maintain context
        max_length: int = 150,
        min_length: int = 40,
        do_sample: bool = False,
    ) -> Optional[str]:
        """
        Summarize a long text by splitting it into smaller chunks, summarizing each,
        and then combining the summaries into a final summary.
        
        :param text: The long text to summarize.
        :param chunk_size: The maximum number of words per chunk.
        :param overlap: The number of words to overlap between chunks.
        :param max_length: Maximum length for each summary (and final summary).
        :param min_length: Minimum length for each summary (and final summary).
        :param do_sample: Whether to use sampling for summarization.
        :return: A summarized string or None if summarization fails.
        """
        if not text:
            logging.warning("Empty text provided for long text summarization.")
            return None

        chunks = self._split_text(text, chunk_size, overlap)
        chunk_summaries: List[str] = []

        logging.info(f"Splitting text into {len(chunks)} chunks for summarization.")
        for idx, chunk in enumerate(chunks):
            logging.info(f"Summarizing chunk {idx + 1}/{len(chunks)}...")
            summary = self.summarize(chunk, max_length=max_length, min_length=min_length, do_sample=do_sample)
            if summary:
                chunk_summaries.append(summary)
            else:
                logging.error(f"Failed to summarize chunk {idx + 1}.")
                return None

        # If there's only one chunk, return its summary; otherwise, summarize the concatenated summaries.
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]
        else:
            combined_text = " ".join(chunk_summaries)
            logging.info("Summarizing combined chunk summaries for final output...")
            final_summary = self.summarize(combined_text, max_length=max_length, min_length=min_length, do_sample=do_sample)
            return final_summary

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split the text into overlapping chunks based on word count.
        
        :param text: The input text to split.
        :param chunk_size: The number of words per chunk.
        :param overlap: The number of overlapping words between chunks.
        :return: A list of text chunks.
        """
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start = end - overlap  # move back by 'overlap' words to maintain context
        return chunks
