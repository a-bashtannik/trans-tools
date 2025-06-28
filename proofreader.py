#!/usr/bin/env python3
"""
Markdown Proofreader with Batch Processing and HTML Output

This script proofreads translated markdown files using OpenAI API with batch processing.
It extracts content from numbered .ru.md files, wraps them in <div class="page"> tags,
and sends them for proofreading in configurable batches. Output is generated as HTML.

Usage:
    python proofreader.py <input_directory> [--system-prompt-file PROMPT_FILE] [--batch BATCH_SIZE] [--pages PAGES] [--output OUTPUT_FILE]

Arguments:
    input_directory: Directory containing numbered .ru.md files (1.ru.md, 2.ru.md, etc.)
    --system-prompt-file: Path to file containing the system prompt for proofreading (default: prompt.enhance.txt)
    --batch: Number of pages to process in each batch (default: 25)
    --pages: Page range to proofread (examples: "5", "1,3,5", "1-5", "1-3,7,10-12")
    --output: Output HTML file for proofreaded content (default: proofreaded_pages.html)
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam
)
from dotenv import load_dotenv


class MarkdownProofreader:
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = None, debug: bool = False):
        """
        Initialize the proofreader.

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use for proofreading
            base_url: Optional base URL for OpenAI API
            debug: Enable debug logging
        """
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.debug = debug

    def load_system_prompt(self, prompt_file_path: str) -> str:
        """Load system prompt from file."""
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"System prompt file not found: {prompt_file_path}")
        except Exception as e:
            raise Exception(f"Error reading system prompt file: {e}")

    def parse_pages(self, page_range: str) -> List[int]:
        """Parse page range string into list of page numbers."""
        if not page_range:
            return []

        pages = set()
        for part in page_range.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                pages.update(range(start, end + 1))
            else:
                pages.add(int(part))
        return sorted(list(pages))

    def get_translated_md_files(self, directory: str, target_pages: List[int] = None) -> List[Tuple[int, Path]]:
        """Get numbered .ru.md files, optionally filtered by target pages."""
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Find all numbered .ru.md files
        md_files = []
        for file_path in directory_path.glob("*.ru.md"):
            match = re.match(r'^(\d+)\.ru\.md$', file_path.name)
            if match:
                file_number = int(match.group(1))
                if not target_pages or file_number in target_pages:
                    md_files.append((file_number, file_path))

        if not md_files:
            raise ValueError(f"No .ru.md files found in {directory}")

        # Sort by file number
        md_files.sort(key=lambda x: x[0])
        return md_files

    def read_full_content(self, file_path: Path) -> str:
        """Read full markdown file content preserving all sections."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {e}")

    def create_batch_content(self, pages_data: List[Tuple[int, str]]) -> str:
        """Create batch content with page wrappers for HTML output."""
        batch_parts = []

        for page_num, full_content in pages_data:
            page_wrapper = f'<div class="page" data-page="{page_num}">\n{full_content}\n</div>'
            batch_parts.append(page_wrapper)

        return '\n\n'.join(batch_parts)

    def clean_ai_response(self, response_content: str) -> str:
        """Clean AI response from markdown code blocks."""
        # Remove ```html and ``` wrapper if present
        cleaned = response_content.strip()

        # Check if response starts with ```html and ends with ```
        if cleaned.startswith('```html'):
            cleaned = cleaned[7:]  # Remove ```html
        elif cleaned.startswith('```'):
            cleaned = cleaned[3:]  # Remove ```

        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]  # Remove closing ```

        return cleaned.strip()

    def proofread_batch(self, batch_content: str, system_prompt: str, batch_num: int, page_numbers: List[int]) -> str:
        """Proofread a batch of pages using OpenAI API."""
        user_message = f"""Proofread and improve the following Russian memoir pages, converting ALL markdown syntax to proper HTML:

{batch_content}"""

        try:
            pages_str = ', '.join(map(str, page_numbers))
            print(f"Proofreading batch {batch_num} (pages: {pages_str})...")

            if self.debug:
                print("\n" + "=" * 80)
                print("DEBUG: SYSTEM PROMPT")
                print("=" * 80)
                print(system_prompt)
                print("\n" + "=" * 80)
                print("DEBUG: USER MESSAGE")
                print("=" * 80)
                print(user_message)
                print("=" * 80 + "\n")

            system_message: ChatCompletionSystemMessageParam = {
                "role": "system",
                "content": system_prompt
            }
            user_message_param: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": user_message
            }

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[system_message, user_message_param],
                temperature=0.25  # Lowered for more consistent output
            )

            proofreaded_content = response.choices[0].message.content

            # Clean the response from markdown code blocks
            proofreaded_content = self.clean_ai_response(proofreaded_content)

            if self.debug:
                print("=" * 80)
                print("DEBUG: RESPONSE")
                print("=" * 80)
                print(proofreaded_content)
                print("=" * 80 + "\n")

            return proofreaded_content

        except Exception as e:
            raise Exception(f"Error proofreading batch {batch_num}: {e}")

    def create_batches(self, files_data: List[Tuple[int, Path]], batch_size: int) -> List[List[Tuple[int, Path]]]:
        """Split files into batches."""
        batches = []
        for i in range(0, len(files_data), batch_size):
            batch = files_data[i:i + batch_size]
            batches.append(batch)
        return batches

    def proofread_directory(self, input_dir: str, system_prompt_file: str, batch_size: int = 25,
                            page_range: str = None, output_file: str = "proofreaded_pages.html"):
        """Proofread translated markdown files in directory."""
        system_prompt = self.load_system_prompt(system_prompt_file)
        target_pages = self.parse_pages(page_range) if page_range else None

        # Get files to process
        files_data = self.get_translated_md_files(input_dir, target_pages)

        print(f"Found {len(files_data)} files for proofreading")
        if target_pages:
            print(f"Target pages: {target_pages}")
        print(f"Batch size: {batch_size}")

        # Create batches
        batches = self.create_batches(files_data, batch_size)
        print(f"Created {len(batches)} batches")

        all_proofreaded_content = []

        try:
            for batch_num, batch_files in enumerate(batches, 1):
                print(f"\nProcessing batch {batch_num}/{len(batches)}")

                # Load and extract article content for each file in batch
                batch_pages_data = []
                page_numbers = []

                for page_num, file_path in batch_files:
                    print(f"  Loading content from {file_path.name}")
                    full_content = self.read_full_content(file_path)
                    batch_pages_data.append((page_num, full_content))
                    page_numbers.append(page_num)

                # Create batch content
                batch_content = self.create_batch_content(batch_pages_data)

                # Proofread batch
                proofreaded_batch = self.proofread_batch(
                    batch_content, system_prompt, batch_num, page_numbers
                )

                all_proofreaded_content.append(proofreaded_batch)
                print(f"  ✅ Batch {batch_num} proofreaded successfully")

            # Write combined output as HTML
            output_path = Path(input_dir) / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write HTML structure
                f.write('<!DOCTYPE html>\n')
                f.write('<html lang="ru">\n')
                f.write('<head>\n')
                f.write('    <meta charset="UTF-8">\n')
                f.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
                f.write('    <title>Proofreaded Pages</title>\n')
                f.write('    <style>\n')
                f.write('        .page { margin-bottom: 2rem; padding: 1rem; border-bottom: 1px solid #ccc; page-break-after: always; }\n')
                f.write('        .page:last-child { page-break-after: avoid; }\n')
                f.write('        article { margin-bottom: 1rem; }\n')
                f.write('        footer { font-style: italic; color: #666; }\n')
                f.write('        img { max-width: 100%; max-height: 50%; height: auto; width: auto; display: block; }\n')
                f.write('        figcaption { font-style: italic; color: #666; margin-top: 0.5rem; text-align: center; }\n')
                f.write('        @media print {\n')
                f.write('            .page { page-break-after: always; margin-bottom: 0; }\n')
                f.write('            .page:last-child { page-break-after: avoid; }\n')
                f.write('            img { max-height: 100vh; page-break-inside: avoid; }\n')
                f.write('        }\n')
                f.write('    </style>\n')
                f.write('</head>\n')
                f.write('<body>\n')

                # Write all proofreaded content
                combined_content = '\n\n'.join(all_proofreaded_content)
                f.write(combined_content)

                f.write('\n</body>\n')
                f.write('</html>\n')

            print(f"\n🎉 Proofreading completed successfully!")
            print(f"📄 Result saved to: {output_path}")

        except Exception as e:
            print(f"❌ Error during proofreading: {e}")
            raise


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Proofread translated markdown files using OpenAI API with batch processing and HTML output"
    )
    parser.add_argument("input_directory", help="Directory containing numbered .ru.md files")
    parser.add_argument("--system-prompt-file", default="prompt.enhance.txt",
                        help="Path to system prompt file (default: prompt.enhance.txt)")
    parser.add_argument("--batch", type=int, default=25,
                        help="Number of pages to process in each batch (default: 25)")
    parser.add_argument("--pages",
                        help="Page range to proofread (examples: '5', '1,3,5', '1-5', '1-3,7,10-12')")
    parser.add_argument("--output", default="proofreaded_pages.html",
                        help="Output file for proofreaded content (default: proofreaded_pages.html)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging to see prompts and responses")

    args = parser.parse_args()

    api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('OPENAI_MODEL', 'gpt-4o')
    base_url = os.getenv('OPENAI_BASE_URL')

    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        sys.exit(1)

    try:
        proofreader = MarkdownProofreader(
            api_key=api_key,
            model=model,
            base_url=base_url,
            debug=args.debug
        )

        proofreader.proofread_directory(
            args.input_directory,
            args.system_prompt_file,
            args.batch,
            args.pages,
            args.output
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()