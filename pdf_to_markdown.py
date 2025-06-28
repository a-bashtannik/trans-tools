#!/usr/bin/env python3
"""
PDF to Markdown converter using Marker with OCR support
Converts all pages of a PDF to separate numbered markdown files
Supports advanced OCR features and language configuration
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from dotenv import load_dotenv
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict


class PDFToMarkdownConverter:
    def __init__(self,
                 languages: Optional[List[str]] = None,
                 force_ocr: bool = False,
                 use_llm: bool = False):
        """
        Initialize the converter with OCR configuration

        Args:
            force_ocr: Force OCR on all lines even for digital PDFs
            use_llm: Use LLM for enhanced conversion quality
        """
        load_dotenv()

        # Get configuration from .env

        openai_api_key = os.getenv('OPENAI_API_KEY')
        openai_model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

        if use_llm and not openai_api_key:
            print("Warning: OPENAI_API_KEY not found in .env file, disabling LLM features")
            use_llm = False
        elif use_llm:
            print(f"Using OpenAI model: {openai_model}")
            # Enable LLM features in Marker
            os.environ['USE_LLM'] = 'true'

        # Configure OCR settings through environment variables
        if force_ocr:
            os.environ['OCR_ALL_PAGES'] = 'true'
            print("Force OCR enabled - will OCR all pages")

        # Store configuration

        self.force_ocr = force_ocr
        self.use_llm = use_llm

        try:
            print("Initializing PDF converter with OCR support...")

            # Create configuration dict for LLM if needed
            config_dict = {}

            if use_llm and openai_api_key:
                config_dict['use_llm'] = True
                config_dict['llm_service'] = 'marker.services.openai.OpenAIService'
                config_dict['openai_api_key'] = openai_api_key
                config_dict['openai_model'] = openai_model
                config_dict['openai_base_url'] = 'https://api.openai.com/v1'

            config_dict['disable_ocr_math'] = True

            # Initialize converter with configuration

            config_parser = ConfigParser(config_dict)

            self.converter = PdfConverter(
                artifact_dict=create_model_dict(),
                config=config_parser.generate_config_dict(),
                llm_service=config_parser.get_llm_service() if use_llm else None
            )

            print("PDF converter initialized successfully!")

        except Exception as e:
            print(f"Error initializing converter: {e}")
            raise

    def get_page_count(self, pdf_path: str) -> int:
        """
        Get the total number of pages in the PDF

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Number of pages in the PDF
        """
        try:
            pdf_document = fitz.open(pdf_path)
            page_count = len(pdf_document)
            pdf_document.close()
            return page_count
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

    def create_single_page_pdf(self, pdf_path: str, page_number: int) -> bytes:
        """
        Create a single-page PDF in memory

        Args:
            pdf_path: Path to the original PDF file
            page_number: Page number to extract (1-based)

        Returns:
            PDF bytes for single page
        """
        try:
            # Open the PDF
            pdf_document = fitz.open(pdf_path)

            # Check if page number is valid
            if page_number < 1 or page_number > len(pdf_document):
                raise ValueError(f"Page {page_number} not found. PDF has {len(pdf_document)} pages.")

            # Create new PDF with single page (convert to 0-based index)
            new_pdf = fitz.open()
            new_pdf.insert_pdf(pdf_document, from_page=page_number - 1, to_page=page_number - 1)

            # Get PDF bytes
            pdf_bytes = new_pdf.tobytes()

            # Close documents
            new_pdf.close()
            pdf_document.close()

            return pdf_bytes

        except Exception as e:
            raise Exception(f"Error creating single page PDF for page {page_number}: {str(e)}")

    def convert_page_to_markdown(self, pdf_path: str, page_number: int, output_dir: Path) -> str:
        """
        Convert a specific page from PDF to Markdown using Marker with OCR

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number to convert (1-based)
            output_dir: Directory where to save the markdown file and images

        Returns:
            Markdown content as string
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            # Create single page PDF in memory
            print(f"Processing page {page_number} from {pdf_path.name}...")
            single_page_bytes = self.create_single_page_pdf(str(pdf_path), page_number)

            # Create a temporary file with unique name for this page
            temp_pdf_path = pdf_path.parent / f"temp_page_{page_number}_{os.getpid()}.pdf"

            # Write single page to temporary file
            with open(temp_pdf_path, 'wb') as f:
                f.write(single_page_bytes)

            # Convert to markdown using Marker with OCR
            print(f"Converting page {page_number} to Markdown...")
            if self.force_ocr:
                print(f"  - Force OCR enabled for page {page_number}")

            markdown_output = self.converter(str(temp_pdf_path))

            # Extract markdown content from output object

            if not hasattr(markdown_output, 'markdown'):
                raise ValueError(f"Conversion failed for page {page_number} - no markdown content found")

            markdown_content = markdown_output.markdown

            # Check if images were extracted and save them with page-specific names
            if hasattr(markdown_output, 'images') and markdown_output.images:
                print(f"Found {len(markdown_output.images)} images on page {page_number} - filtering and saving them...")

                # Create images directory
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save each image with page-specific naming and size filtering
                saved_images = 0
                for image_name, image_data in markdown_output.images.items():
                    try:
                        # Get image dimensions
                        width, height = None, None

                        if hasattr(image_data, 'size'):
                            # PIL Image object
                            width, height = image_data.size
                        elif isinstance(image_data, bytes):
                            # Raw bytes - need to load as PIL to get dimensions
                            from PIL import Image
                            import io
                            img = Image.open(io.BytesIO(image_data))
                            width, height = img.size
                        else:
                            # Try to convert to PIL
                            from PIL import Image
                            import io
                            img = Image.open(io.BytesIO(image_data))
                            width, height = img.size

                        # Filter out small images (artifacts)
                        if width and height and (width < 170 or height < 170):
                            print(f"Skipping small image {image_name} ({width}x{height}px) - likely an artifact")
                            # Remove reference to this image from markdown content
                            markdown_content = markdown_content.replace(f"![{image_name}]({image_name})", "")
                            markdown_content = markdown_content.replace(f"![]({image_name})", "")
                            markdown_content = markdown_content.replace(image_name, "")
                            continue

                        # Add page number prefix to image name to avoid conflicts
                        base_name, ext = os.path.splitext(image_name)
                        page_specific_name = f"page_{page_number}_{base_name}{ext}"
                        image_path = output_dir / page_specific_name

                        # Update markdown content to use the new image name
                        markdown_content = markdown_content.replace(image_name, page_specific_name)

                        # Save image data
                        if hasattr(image_data, 'save'):
                            # PIL Image object
                            image_data.save(image_path)
                        elif isinstance(image_data, bytes):
                            # Raw bytes
                            with open(image_path, 'wb') as f:
                                f.write(image_data)
                        else:
                            # Convert to bytes and save
                            with open(image_path, 'wb') as f:
                                f.write(image_data)

                        print(f"Saved image: {image_path} ({width}x{height}px)")
                        saved_images += 1

                    except Exception as e:
                        print(f"Error processing image {image_name}: {e}")

                print(f"Saved {saved_images} images (filtered out small artifacts)")
            else:
                print(f"No images found on page {page_number}")

            return markdown_content

        except Exception as e:
            raise Exception(f"Error converting page {page_number} to Markdown: {str(e)}")

        finally:
            # Clean up temporary file
            if 'temp_pdf_path' in locals() and temp_pdf_path.exists():
                try:
                    temp_pdf_path.unlink()
                except:
                    pass  # Ignore cleanup errors

    def convert_all_pages_to_markdown(self, pdf_path: str, output_dir: str, merge: bool = False):
        """
        Convert all pages from PDF to Markdown files with OCR
    
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory where to save markdown files and images
            merge: If True, save all pages as one single file instead of separate files
        """
        pdf_path_obj = Path(pdf_path)
        output_dir_obj = Path(output_dir)
    
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
        # Create output directory
        output_dir_obj.mkdir(parents=True, exist_ok=True)
    
        # Get total number of pages
        total_pages = self.get_page_count(pdf_path)
        print(f"PDF has {total_pages} pages. Converting all pages...")
    
        # Print OCR configuration
        print(f"OCR Configuration:")
        print(f"  - Force OCR: {self.force_ocr}")
        print(f"  - Use LLM: {self.use_llm}")
        print(f"  - Merge pages: {merge}")
    
        # Convert each page sequentially
        success_count = 0
        all_markdown_content = []
        
        for page_number in range(1, total_pages + 1):
            try:
                print(f"\n--- Converting page {page_number}/{total_pages} ---")
    
                # Convert page to markdown
                markdown_content = self.convert_page_to_markdown(
                    pdf_path,
                    page_number,
                    output_dir_obj
                )
                
                if merge:
                    # Add page number header and content to merged document
                    page_header = f"\n\n## Page {page_number}\n\n"
                    all_markdown_content.append(page_header + markdown_content)
                    print(f"Added page {page_number} content to merged document")
                else:
                    # Create output filename for individual page
                    output_filename = f"{page_number}.md"
                    output_path = output_dir_obj / output_filename
    
                    # Save markdown file
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
    
                    print(f"Saved: {output_path}")
                
                success_count += 1
    
            except Exception as e:
                print(f"Error converting page {page_number}: {str(e)}")
                continue
    
        # If merging pages, save the combined content to a single file
        if merge and all_markdown_content:
            # Get base filename without extension
            base_filename = pdf_path_obj.stem
            merged_filename = f"{base_filename}.md"
            merged_path = output_dir_obj / merged_filename
            
            # Add document title as H1 heading
            document_title = f"# {base_filename}\n\n"
            merged_content = document_title + "".join(all_markdown_content)
            
            # Save merged markdown file
            with open(merged_path, 'w', encoding='utf-8') as f:
                f.write(merged_content)
                
            print(f"\nSaved merged document: {merged_path}")
    
        print(f"\n✅ Conversion complete! {success_count}/{total_pages} pages successfully converted")
        print(f"📁 Output saved to: {output_dir_obj}")
        
        if merge:
            print(f"📄 Merged output saved as: {merged_filename}")
        else:
            print(f"💡 Files are named as 1.md, 2.md, 3.md, etc.")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown with OCR support using Marker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_to_markdown.py document.pdf ./output/
  python pdf_to_markdown.py --force-ocr document.pdf ./output/
  python pdf_to_markdown.py --use-llm document.pdf ./output/
      python pdf_to_markdown.py --merge document.pdf ./output/
"""
    )

    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("output_dir", help="Directory to save markdown files")

    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR on all pages, even for digital PDFs"
    )

    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for enhanced conversion quality (requires OPENAI_API_KEY)"
    )

    parser.add_argument(
        "--merge",
        action="store_true",
        help="Save all pages as a single markdown file instead of separate files"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the converter"""
    args = parse_arguments()

    languages = ['pl']

    try:
        print("🚀 Starting PDF to Markdown conversion with OCR support...")
        print(f"📄 Input: {args.pdf_path}")
        print(f"📁 Output: {args.output_dir}")
        print(f"🌍 Languages: {', '.join(languages)}")
        if args.merge:
            print(f"📎 Merge mode: ON - all pages will be saved as a single file")

        # Initialize converter with OCR configuration
        converter = PDFToMarkdownConverter(
            languages=languages,
            force_ocr=args.force_ocr,
            use_llm=args.use_llm
        )

        # Convert all pages to markdown
        converter.convert_all_pages_to_markdown(args.pdf_path, args.output_dir, merge=args.merge)

        print(f"\n🎉 All pages successfully converted from {args.pdf_path}")
        print(f"📁 Output saved to: {args.output_dir}")
        if not args.merge:
            print(f"💡 Files are named as 1.md, 2.md, 3.md, etc.")
        print(f"✨ Math processing disabled - no LaTeX formulas generated")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
