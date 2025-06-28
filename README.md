# PDF Processing Toolkit

A Python toolkit for converting PDFs to translated content with OCR support, cleaning, and proofreading. Designed for processing documents like memoirs, books, or academic papers that need accurate translation.

## Setup

Install dependencies:
```bash
pip install python-dotenv openai pathlib fitz pymupdf marker-pdf
```

Create `.env` file:
```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o
```

Create prompt files: `prompt.txt` for translation, `prompt.enhance.txt` for proofreading.

## Processing Flow

```
PDF → Extract Pages → Clean Content → Fix Hyphens → Translate → Proofread → HTML
```

**pdf_to_markdown.py** - Converts PDF pages to numbered markdown files using OCR. Extracts images and handles both digital and scanned PDFs. Filters images smaller than 170x170px to remove artifacts.

**cleaner.py** - Removes OCR artifacts, LaTeX formulas (`$...$` patterns), and page numbers. Organizes content into `<article>` and `<footer>` sections based on `<sup>` tag detection.

**hyphen_fix.py** - Fixes words broken across pages by hyphens. Analyzes sentence context and uses AI to select correct word continuation from next 3 files. Moves sentence fragments between files when needed.

**markdown_translator.py** - Translates content using sliding context window algorithm. Includes previous N pages as context in each translation request to maintain consistency across the document.

**proofreader.py** - Proofreads translated content in configurable batches. Wraps each page in `<div class="page">` tags and processes multiple pages in single API calls for efficiency.

## Extract Pages - pdf_to_markdown.py

Converts PDF pages to numbered markdown files using the Marker library with OCR support.

**Purpose**: Extract text and images from PDF documents, handling both digital and scanned content.

**Algorithm**:
1. **Page Extraction**: Creates single-page PDF for each page using PyMuPDF
2. **OCR Processing**: Uses Marker library to convert PDF to markdown with configurable OCR settings
3. **Image Filtering**: Extracts all images, then filters out artifacts smaller than 170x170px
4. **Image Naming**: Prefixes remaining images with `page_N_` to prevent filename conflicts
5. **Content Cleanup**: Removes markdown references to filtered images from text

**Usage**:
```bash
# Basic conversion
python pdf_to_markdown.py memoir.pdf ./output/

# Force OCR for scanned documents
python pdf_to_markdown.py memoir.pdf ./output/ --force-ocr

# Enhanced AI processing
python pdf_to_markdown.py memoir.pdf ./output/ --force-ocr --use-llm

# Merge all pages into single file
python pdf_to_markdown.py memoir.pdf ./output/ --merge
```

**Output**: Creates numbered files `1.md`, `2.md`, `3.md`, etc. with extracted images in the same directory.

## Clean Content - cleaner.py

Removes OCR artifacts and organizes content into structured sections.

**Purpose**: Clean up OCR output and standardize document structure.

**Algorithm**:
1. **Fragment Removal**: Removes hardcoded patterns like `"&</sup>lt;sup>"` from FRAGMENTS_TO_REMOVE list
2. **LaTeX Cleanup**: Removes mathematical formulas matching `$...$` pattern using regex
3. **Content Sectioning**: 
   - Finds first line starting with `<sup>` tag
   - Splits content: everything before = main content, everything after = footnotes
   - Wraps main content in `<article>` tags
   - Wraps footnotes in `<footer>` tags
4. **Page Number Removal**: Detects and removes trailing page numbers (digits or `$digits$`)
5. **Whitespace Trimming**: Removes empty lines and trailing spaces inside article sections

**Usage**:
```bash
# Clean all markdown files in directory
python cleaner.py ./output/

# Process specific directory
python cleaner.py /path/to/markdown/files/
```

**Example transformation**:
```
Before: "Some text $x^2 + y = z$ more text&</sup>lt;sup>1</sup> footnote"
After:  "<article>Some text more text</article>---<footer><sup>1</sup> footnote</footer>"
```

## Fix Hyphens - hyphen_fix.py

Fixes words broken across page boundaries by hyphens.

**Purpose**: Reconstruct words split by page breaks while maintaining document flow.

**Algorithm**:
1. **Hyphen Detection**: Scans all files to find `<article>` sections ending with hyphen
2. **Candidate Analysis**: For each hyphen file, examines next 3 files for potential continuations
3. **Context Extraction**: Shows user/AI the original phrase ending with hyphen
4. **Decision Making**: 
   - Manual mode: User selects correct continuation file
   - AI mode: Sends options to LLM for automatic selection
5. **Sentence Boundary Detection**: Finds sentence start by scanning backwards for `.`, `?`, `!`
6. **Content Transfer**: 
   - Extracts sentence fragment from source file
   - Combines with first word from target file
   - Updates both files with reconstructed content

**Usage**:
```bash
# Manual review mode
python hyphen_fix.py ./output/

# Automatic AI mode
python hyphen_fix.py ./output/ --use-llm
```

**Example fix**:
- **Before**: File 5.md ends with "przykład-", File 6.md starts with "owo"
- **After**: File 5.md ends with "przykład", File 6.md starts with "przykładowo"

## Translate - markdown_translator.py

Translates content using sliding context window for consistency.

**Purpose**: Provide contextually accurate translations by considering surrounding pages.

**Context Window Algorithm**:
1. **Window Construction**: For page N, includes pages N-P to N-1 as context (P = window size)
2. **Bidirectional Mode**: When enabled, also includes pages N+1 to N+P
3. **Context Formatting**: Labels context as "Previous Page -1", "Previous Page -2", etc.
4. **Translation Request**: Combines context + current page in single API call
5. **Consistency Maintenance**: AI uses context to maintain terminology and style consistency

**Usage**:
```bash
# Basic translation to Russian
python markdown_translator.py ./output/ --lang ru

# With context window
python markdown_translator.py ./output/ --lang ru --context-window 2

# Bidirectional context
python markdown_translator.py ./output/ --lang ru --bidirectional

# Specific pages
python markdown_translator.py ./output/ --lang ru --pages "1-10,15,20-25"
```

**Context Example** (context-window 2):
- Page 5: Uses pages 3-4 as context
- Page 6: Uses pages 4-5 as context  
- Bidirectional: Page 5 uses pages 3-4 and 6-7

**Output**: Creates translated files `1.ru.md`, `2.ru.md`, `3.ru.md`, etc.

## Proofread - proofreader.py

Proofreads translated content in batches and generates HTML output.

**Purpose**: Improve translation quality and convert to publication-ready format.

**Batch Processing Algorithm**:
1. **File Collection**: Gathers all `.ru.md` files in numerical order
2. **Batch Creation**: Groups consecutive pages into batches of specified size
3. **Page Wrapping**: Wraps each page content in `<div class="page" data-page="N">` tags
4. **Batch Assembly**: Combines all pages in batch into single request
5. **API Optimization**: Processes entire batch in one API call instead of individual calls
6. **Response Processing**: Extracts and reassembles individual page results
7. **HTML Generation**: Creates complete HTML document with CSS styling

**Usage**:
```bash
# Basic proofreading
python proofreader.py ./output/

# Custom batch size
python proofreader.py ./output/ --batch 10

# Specific pages and output file
python proofreader.py ./output/ --pages "1-50" --output part1.html
```

**Efficiency Example** (batch size 10):
- 100 pages = 10 API calls instead of 100
- Faster processing and lower API costs
- Maintains formatting consistency within batches

**Output**: Clean HTML file ready for printing or digital reading.

## Complete Usage Example

Process a Polish memoir to Russian:

```bash
# 1. Extract PDF pages
python pdf_to_markdown.py memoir.pdf ./output/ --force-ocr

# 2. Clean content
python cleaner.py ./output/

# 3. Fix broken words
python hyphen_fix.py ./output/ --use-llm

# 4. Translate with context
python markdown_translator.py ./output/ --lang ru --context-window 2

# 5. Proofread and generate HTML
python proofreader.py ./output/ --output memoir_final.html
```

## Reviewing with Obsidian

Use Obsidian to review the markdown files. Point it to your output directory as a vault. The file explorer shows all numbered pages, and you can compare original and translated versions side by side in split view.

## Troubleshooting

**Poor OCR quality**: Add `--use-llm` to PDF conversion

**Broken translations**: Increase `--context-window` or add `--bidirectional`

**Memory issues**: Use smaller `--pages` ranges or reduce `--batch` size

**Debug mode**: Add `--debug` to any tool to see AI prompts and responses