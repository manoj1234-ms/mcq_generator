# TODO: Optimize MCQ Generator with Parallel Chunk Processing and Previous Data Usage

## Tasks
- [x] Import concurrent.futures in app.py for parallel processing
- [x] Modify generate_mcqs_with_fallback to use ThreadPoolExecutor with limited workers (e.g., 3) to process chunks in parallel
- [x] Adjust rate limiting: reduce delays since parallel, but add semaphore if needed
- [x] Implement caching for generated MCQs based on file hash to reuse previous results
- [x] Modify prompt to include context from previous chunks for better answer generation
- [x] Test the parallel processing and caching functionality
