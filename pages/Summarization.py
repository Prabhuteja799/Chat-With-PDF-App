import streamlit as st
import openai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
from typing import List, Tuple
import time

class PDFSummarizer:
    def __init__(self):
        """Initialize the PDFSummarizer with default settings"""
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            st.error("OpenAI API key not found. Please set it in your .env file.")
            st.stop()
        openai.api_key = self.openai_api_key

    @staticmethod
    def read_pdf(file) -> Tuple[str, int]:
        """
        Read PDF and return text content and total word count
        """
        try:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            word_count = len(text.split())
            return text, word_count
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return "", 0

    @staticmethod
    def calculate_suggested_summary_length(word_count: int) -> Tuple[int, int, int]:
        """
        Calculate suggested summary length based on document size
        Returns min_length, max_length, default_length
        """
        # Roughly estimate that summary should be 10-20% of original text
        min_length = max(100, int(word_count * 0.1))
        max_length = max(500, int(word_count * 0.2))
        default_length = (min_length + max_length) // 2
        
        # Cap the values to reasonable limits
        min_length = min(min_length, 1000)
        max_length = min(max_length, 2000)
        default_length = min(default_length, 1500)
        
        return min_length, max_length, default_length

    def split_text(self, text: str, max_chunk_size: int = 2000) -> List[str]:
        """
        Split text into chunks while preserving sentence boundaries
        """
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks

    def generate_summary(self, text: str, target_length: int, format_type: str = "paragraph") -> str:
        """
        Generate summary using OpenAI API with better prompt engineering
        """
        try:
            prompt = f"""Please provide a {format_type} summary of the following text. 
                The summary should be approximately {target_length} words long and capture the key points 
                while maintaining coherence and readability. Focus on the main ideas and essential details.

                Text: {text}"""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a precise summarization assistant. Create clear, "
                                                  "coherent summaries that capture key information while maintaining "
                                                  "the requested length."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=int(target_length * 1.5),  # Allow some buffer for natural language
                temperature=0.5,  # Balance between creativity and consistency
                presence_penalty=0.1,  # Slight penalty for repetition
                frequency_penalty=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")
            return ""

def main():
    st.set_page_config(
        page_title="PDF Summarizer",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚PDF Summarizer")
    st.markdown("""
    Upload your PDF document to generate a concise, intelligent summary.
    The summary length will be automatically suggested based on your document's size.
    """)

    summarizer = PDFSummarizer()

    # File upload with better UX
    uploaded_file = st.file_uploader(
        "Upload your PDF file (Max 10MB)",
        type=["pdf"],
        help="Upload a PDF document to generate its summary"
    )

    if uploaded_file is not None:
        # Check file size
        if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
            st.error("File size exceeds 10MB limit. Please upload a smaller file.")
            st.stop()

        # Read PDF with progress indicator
        with st.spinner("Reading PDF..."):
            text, word_count = summarizer.read_pdf(uploaded_file)

        if text:
            st.success(f"Successfully read PDF. Document contains {word_count:,} words.")

            # Calculate suggested summary lengths
            min_length, max_length, default_length = summarizer.calculate_suggested_summary_length(word_count)

            # Summary settings  
            summary_length = st.slider(
                    "Target Summary Length (words):",
                    min_value=min_length,
                    max_value=max_length,
                    value=default_length,
                    step=50,
                    help=f"Suggested length based on document size: {default_length} words"
                )

            # Generate summary with progress tracking
            if st.button("Generate Summary", help="Click to generate the summary"):
                with st.spinner("Generating summary..."):
                    # Split text into chunks and process
                    chunks = summarizer.split_text(text)
                    summaries = []
                    progress_bar = st.progress(0)
                    
                    for i, chunk in enumerate(chunks):
                        chunk_summary = summarizer.generate_summary(
                            chunk,
                            max(100, summary_length // len(chunks)),
                            
                        )
                        if chunk_summary:
                            summaries.append(chunk_summary)
                        progress_bar.progress((i + 1) / len(chunks))
                        time.sleep(0.1)  # Smooth progress bar animation

                    # Combine and display final summary
                    if summaries:
                        final_summary = " ".join(summaries)
                        st.header("📝 Summary")
                        st.write(final_summary)
                        
                        # Summary statistics
                        summary_word_count = len(final_summary.split())
                        st.info(f"""
                        Summary Statistics:
                        - Original document: {word_count:,} words
                        - Summary length: {summary_word_count:,} words
                        - Compression ratio: {(summary_word_count/word_count*100):.1f}%
                        """)
                        
                        # Download button for summary
                        st.download_button(
                            label="Download Summary",
                            data=final_summary,
                            file_name="summary.pdf",
                            mime="text/plain"
                        )
                    else:
                        st.error("Failed to generate summary. Please try again.")

if __name__ == "__main__":
    main()