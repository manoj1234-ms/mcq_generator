import os
import time
import random
import re
from flask import Flask, render_template, request, send_file
import pdfplumber
import docx
import csv
from werkzeug.utils import secure_filename
import google.generativeai as genai
from fpdf import FPDF  # pip install fpdf

# Set your API key from environment variable
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY environment variable not set")
genai.configure(api_key=api_key)

# Initialize models with fallback strategy
flash_model = genai.GenerativeModel("models/gemini-2.5-flash")
pro_model = genai.GenerativeModel("models/gemini-2.5-pro")

# Configuration for API limits and reliability
API_CONFIG = {
    'max_retries': 3,
    'base_delay': 2,  # Increased base delay
    'max_chunk_size': 6000,  # Reduced chunk size to avoid token limits
    'max_questions': 30,  # Reduced to be more conservative with free tier
    'min_text_length': 100,
    'rate_limit_delay': 3  # Delay between chunk processing
}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULTS_FOLDER'] = 'results/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'docx'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def split_text_into_chunks(text, max_chunk_size=None):
    """Split text into chunks for better processing"""
    if max_chunk_size is None:
        max_chunk_size = API_CONFIG['max_chunk_size']
    
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        if current_size + len(word) + 1 > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def call_gemini_with_retry(model, prompt, max_retries=None, base_delay=None):
    """Call Gemini API with exponential backoff retry logic"""
    if max_retries is None:
        max_retries = API_CONFIG['max_retries']
    if base_delay is None:
        base_delay = API_CONFIG['base_delay']
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
            else:
                raise Exception("Empty response from API")
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                # Check if the error contains retry_delay information
                retry_delay = base_delay * (2 ** attempt)
                
                # Try to extract retry_delay from the error message
                error_str = str(e)
                if "retry_delay" in error_str:
                    try:
                        # Extract the retry_delay value from the error
                        delay_match = re.search(r'seconds: (\d+)', error_str)
                        if delay_match:
                            retry_delay = int(delay_match.group(1))
                            print(f"Using API-suggested retry delay: {retry_delay} seconds")
                    except:
                        pass
                
                # Add jitter to prevent thundering herd
                delay = retry_delay + random.uniform(0, 2)
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                raise e
    
    return None

import concurrent.futures
import threading
import hashlib

# Semaphore to limit concurrent API calls to avoid rate limit issues
api_semaphore = threading.Semaphore(3)

# Simple in-memory cache for generated MCQs
mcq_cache = {}

def generate_mcqs_for_chunk(i, chunk, num_questions_per_chunk, previous_context=""):
    """Generate MCQs for a single chunk with optional previous context"""
    print(f"Processing chunk {i + 1} with parallel processing")
    
    prompt = f"""
    You are an AI assistant helping generate multiple-choice questions (MCQs) based on the following text:
    '{previous_context} {chunk}'
    
    Please generate exactly {num_questions_per_chunk} MCQs from this text chunk. Each question should have:
    - A clear, well-formulated question
    - Four answer options (labeled A, B, C, D)
    - The correct answer clearly indicated
    - Make questions challenging but fair
    
    Format each MCQ exactly like this:
    ## MCQ
    Question: [question]
    A) [option A]
    B) [option B]
    C) [option C]
    D) [option D]
    Correct Answer: [correct option]
    
    """
    
    with api_semaphore:
        # Try flash model first (faster, cheaper)
        try:
            mcqs = call_gemini_with_retry(flash_model, prompt)
            if mcqs:
                print(f"Chunk {i + 1} processed successfully with Flash model")
                return mcqs
        except Exception as e:
            print(f"Flash model failed for chunk {i + 1}: {str(e)}")
        
        # Fallback to pro model
        try:
            mcqs = call_gemini_with_retry(pro_model, prompt)
            if mcqs:
                print(f"Chunk {i + 1} processed successfully with Pro model")
                return mcqs
            else:
                print(f"Pro model also failed for chunk {i + 1}")
        except Exception as e:
            print(f"Pro model failed for chunk {i + 1}: {str(e)}")
    
    return ""

def generate_mcqs_with_fallback(text_chunks, num_questions_per_chunk):
    """Generate MCQs with model fallback strategy using parallel processing and context"""
    all_mcqs = []
    previous_context = ""
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for i, chunk in enumerate(text_chunks):
            # Include previous context for better answers
            futures.append(executor.submit(generate_mcqs_for_chunk, i, chunk, num_questions_per_chunk, previous_context))
            # Update previous context with current chunk (simple concatenation, can be improved)
            previous_context += " " + chunk
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                all_mcqs.append(result)
    
    return '\n\n'.join(all_mcqs)

def extract_text_from_file(file_obj, filename):
    try:
        ext = filename.rsplit('.', 1)[1].lower()
        if ext == 'pdf':
            with pdfplumber.open(file_obj) as pdf:
                text = ''.join([page.extract_text() for page in pdf.pages if page.extract_text()])
            return text
        elif ext == 'docx':
            doc = docx.Document(file_obj)
            text = ' '.join([para.text for para in doc.paragraphs if para.text.strip()])
            return text
        elif ext == 'txt':
            content = file_obj.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            return content
        return None
    except Exception as e:
        print(f"Error extracting text from {filename}: {str(e)}")
        return None

def Question_mcqs_generator(input_text, num_questions):
    """Enhanced MCQ generator with batching and fallback strategy"""
    print(f"Generating {num_questions} MCQs from text of length {len(input_text)} characters")
    
    # Split text into chunks for better processing
    text_chunks = split_text_into_chunks(input_text)
    print(f"Text split into {len(text_chunks)} chunks")
    
    # Calculate questions per chunk
    questions_per_chunk = max(1, num_questions // len(text_chunks))
    remaining_questions = num_questions % len(text_chunks)
    
    print(f"Questions per chunk: {questions_per_chunk}, Remaining: {remaining_questions}")
    
    # Generate MCQs with fallback strategy
    all_mcqs = generate_mcqs_with_fallback(text_chunks, questions_per_chunk)
    
    if remaining_questions > 0 and text_chunks:
        # Generate remaining questions from the first chunk
        print(f"Generating {remaining_questions} additional questions from first chunk")
        extra_prompt = f"""
        You are an AI assistant helping generate additional multiple-choice questions (MCQs) based on the following text:
        '{text_chunks[0]}'
        
        Please generate exactly {remaining_questions} additional MCQs from this text. Each question should have:
        - A clear, well-formulated question
        - Four answer options (labeled A, B, C, D)
        - The correct answer clearly indicated
        - Make questions challenging but fair
        - Ensure questions are different from any previous ones
        
        Format each MCQ exactly like this:
        ## MCQ
        Question: [question]
        A) [option A]
        B) [option B]
        C) [option C]
        D) [option D]
        Correct Answer: [correct option]
        """
        
        try:
            extra_mcqs = call_gemini_with_retry(flash_model, extra_prompt)
            if extra_mcqs:
                all_mcqs += '\n\n' + extra_mcqs
                print("Additional questions generated successfully")
        except Exception as e:
            print(f"Failed to generate additional questions: {str(e)}")
    
    return all_mcqs

def save_mcqs_to_file(mcqs, filename):
    results_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    with open(results_path, 'w') as f:
        f.write(mcqs)
    return results_path

def create_pdf(mcqs, filename):
    """Create a well-formatted PDF with proper MCQ layout and wrapping"""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Set up fonts and margins
        pdf.set_font("Helvetica", size=11)
        pdf.set_margins(15, 15, 15)  # Left, top, right margins
        
        # Split MCQs by "## MCQ" marker
        mcq_sections = mcqs.split("## MCQ")
        
        for i, section in enumerate(mcq_sections):
            if section.strip():
                # Add MCQ header (skip first empty section)
                if i > 0:
                    pdf.set_font("Helvetica", style="B", size=12)
                    pdf.multi_cell(0, 8, f"MCQ {i}", new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Helvetica", size=11)
                    pdf.ln(2)
                
                # Process the MCQ content
                _format_mcq_section(pdf, section.strip())
                
                # Add space between MCQs
                pdf.ln(8)
                
                # Check if we need a new page
                if pdf.get_y() > 250:  # 250mm is roughly where we want page breaks
                    pdf.add_page()

        pdf_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        pdf.output(pdf_path)
        return pdf_path
        
    except Exception as e:
        print(f"Error creating PDF: {str(e)}")
        raise e

def _format_mcq_section(pdf, section_text):
    """Format a single MCQ section with proper wrapping and structure"""
    lines = section_text.split('\n')
    current_question = ""
    options = []
    correct_answer = ""
    
    # Parse the MCQ structure
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("Question:"):
            current_question = line.replace("Question:", "").strip()
        elif line.startswith("A)") or line.startswith("B)") or line.startswith("C)") or line.startswith("D)"):
            options.append(line)
        elif line.startswith("Correct Answer:"):
            correct_answer = line.replace("Correct Answer:", "").strip()
        elif current_question and not any(line.startswith(x) for x in ["A)", "B)", "C)", "D)", "Correct Answer:"]):
            # This might be a continuation of the question
            current_question += " " + line
    
    # Format and add to PDF
    if current_question:
        # Add question with proper wrapping
        pdf.set_font("Helvetica", style="B", size=11)
        question_text = f"Question: {current_question}"
        pdf.multi_cell(0, 6, question_text, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        
        # Add options with proper wrapping
        pdf.set_font("Helvetica", size=10)
        for option in options:
            # Use multi_cell for proper text wrapping
            pdf.multi_cell(0, 5, option, new_x="LMARGIN", new_y="NEXT")
        
        pdf.ln(3)
        
        # Add correct answer with proper wrapping
        if correct_answer:
            pdf.set_font("Helvetica", style="B", size=10)
            pdf.set_text_color(0, 128, 0)  # Green color for correct answer
            answer_text = f"Correct Answer: {correct_answer}"
            pdf.multi_cell(0, 5, answer_text, new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)  # Reset to black
    else:
        # Fallback: if we can't parse the structure, just add the text with wrapping
        pdf.set_font("Helvetica", size=10)
        # Split long text into manageable chunks
        words = section_text.split()
        current_line = ""
        for word in words:
            if len(current_line + word) > 80:  # Approximate character limit per line
                if current_line:
                    pdf.multi_cell(0, 5, current_line.strip(), new_x="LMARGIN", new_y="NEXT")
                current_line = word + " "
            else:
                current_line += word + " "
        
        if current_line:
            pdf.multi_cell(0, 5, current_line.strip(), new_x="LMARGIN", new_y="NEXT")

def create_simple_pdf(mcqs, filename):
    """Create a simple fallback PDF when the main PDF creation fails"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)
        pdf.set_margins(20, 20, 20)
        
        # Add title
        pdf.set_font("Helvetica", style="B", size=14)
        pdf.multi_cell(0, 8, "Generated MCQs", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
        # Add the text with proper wrapping
        pdf.set_font("Helvetica", size=10)
        lines = mcqs.split('\n')
        for line in lines:
            if line.strip():
                # Use multi_cell for proper text wrapping
                pdf.multi_cell(0, 5, line.strip(), new_x="LMARGIN", new_y="NEXT")
        
        pdf_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        pdf.output(pdf_path)
        return pdf_path
        
    except Exception as e:
        print(f"Simple PDF creation failed: {str(e)}")
        raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_mcqs():
    try:
        if 'file' not in request.files:
            return "No file part", 400

        file = request.files['file']

        if not file or file.filename == '':
            return "No file selected", 400

        if not allowed_file(file.filename):
            return "Invalid file format. Please upload PDF, TXT, or DOCX files only.", 400

        filename = secure_filename(file.filename)

        # Extract text from the uploaded file directly from memory
        text = extract_text_from_file(file, filename)

        if not text:
            return "Could not extract text from the uploaded file. Please check if the file is valid and not corrupted.", 400

        if len(text.strip()) < API_CONFIG['min_text_length']:
            return f"The extracted text is too short to generate meaningful MCQs. Please upload a document with at least {API_CONFIG['min_text_length']} characters.", 400

        user_question = request.form.get('user_question', '').strip()

        if user_question:
            # If user typed a specific question, generate only one MCQ focused on that question
            print(f"Generating MCQ for user question: {user_question}")

            prompt = f"""
            You are an AI assistant helping generate a single multiple-choice question (MCQ) based on the following text:
            '{text}'

            Please generate exactly 1 MCQ that best matches the user's question:
            '{user_question}'

            The MCQ should have:
            - A clear, well-formulated question
            - Four answer options (labeled A, B, C, D)
            - The correct answer clearly indicated
            - Make the question challenging but fair

            Format the MCQ exactly like this:
            ## MCQ
            Question: [question]
            A) [option A]
            B) [option B]
            C) [option C]
            D) [option D]
            Correct Answer: [correct option]
            """

            try:
                mcqs = call_gemini_with_retry(flash_model, prompt)
                if not mcqs:
                    raise Exception("Empty response from API")
                print("MCQ generation for user question completed successfully")
            except Exception as e:
                print(f"MCQ generation for user question failed: {str(e)}")
                return f"Failed to generate MCQ for your question: {str(e)}", 500

            num_questions = 1  # Override to 1 since only one question generated

        else:
            # No user question, proceed with normal batch generation
            try:
                num_questions = int(request.form['num_questions'])
                if num_questions < 1 or num_questions > API_CONFIG['max_questions']:
                    return f"Number of questions must be between 1 and {API_CONFIG['max_questions']}.", 400
            except ValueError:
                return "Invalid number of questions. Please enter a valid number.", 400

            # Check cache
            cache_key = hashlib.md5((text + str(num_questions)).encode()).hexdigest()
            if cache_key in mcq_cache:
                mcqs = mcq_cache[cache_key]
                print("Using cached MCQs")
            else:
                print(f"Starting MCQ generation for {num_questions} questions...")
                try:
                    mcqs = Question_mcqs_generator(text, num_questions)
                    mcq_cache[cache_key] = mcqs  # Cache the result
                    print("MCQ generation completed successfully")
                except Exception as e:
                    print(f"MCQ generation failed: {str(e)}")
                    return f"Failed to generate MCQs: {str(e)}", 500

        if not mcqs or len(mcqs.strip()) < 50:
            return "Failed to generate MCQs. Please try again or upload a different document.", 500

        # Save the generated MCQs to files
        txt_filename = f"generated_mcqs_{filename.rsplit('.', 1)[0]}.txt"
        pdf_filename = f"generated_mcqs_{filename.rsplit('.', 1)[0]}.pdf"
        
        # Save text file
        save_mcqs_to_file(mcqs, txt_filename)
        
        # Create PDF with error handling
        try:
            create_pdf(mcqs, pdf_filename)
            pdf_created = True
        except Exception as e:
            print(f"PDF creation failed: {str(e)}")
            # Create a simple fallback PDF
            try:
                create_simple_pdf(mcqs, pdf_filename)
                pdf_created = True
            except Exception as e2:
                print(f"Fallback PDF creation also failed: {str(e2)}")
                pdf_created = False

        # Display and allow downloading
        is_single_mcq = (num_questions == 1)
        return render_template('results.html',
                            mcqs=mcqs,
                            txt_filename=txt_filename,
                            pdf_filename=pdf_filename if pdf_created else None,
                            is_single_mcq=is_single_mcq)
        
    except Exception as e:
        print(f"Error in generate_mcqs: {str(e)}")
        return f"An error occurred: {str(e)}", 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        if not os.path.exists(file_path):
            return "File not found", 404
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        print(f"Error downloading file {filename}: {str(e)}")
        return "Error downloading file", 500

# For Vercel serverless deployment
def __call__(environ, start_response):
    return app(environ, start_response)

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['RESULTS_FOLDER']):
        os.makedirs(app.config['RESULTS_FOLDER'])
    app.run()
