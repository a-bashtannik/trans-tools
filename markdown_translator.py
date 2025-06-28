#!/usr/bin/env python3
"""
Markdown Translator with Sliding Context Window

This script translates numbered markdown files using OpenAI API with a sliding
context window to maintain translation consistency across files.

Usage:
    python translator.py <input_directory> [--system-prompt-file PROMPT_FILE] [--lang LANG] [--context-window N] [--pages PAGES] [--bidirectional]

Arguments:
    input_directory: Directory containing numbered markdown files (1.md, 2.md, etc.)
    --system-prompt-file: Path to file containing the system prompt for translation (default: prompt.txt)
    --lang: Target language code (default: ru)
    --context-window: Number of previous pages to include as context (default: 2)
    --pages: Page range to translate (examples: "5", "1,3,5", "1-5", "1-3,7,10-12")
    --bidirectional: Include both previous and next pages in context
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import List, Optional
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam
)
from dotenv import load_dotenv


class MarkdownTranslator:
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = None, context_window_size: int = 2, target_lang: str = "ru",
                 bidirectional: bool = False, debug: bool = False):
        """
        Initialize the translator.

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use for translation
            base_url: Optional base URL for OpenAI API
            context_window_size: Number of previous pages to include as context (P parameter)
            target_lang: Target language code for translation
            bidirectional: Include both previous and next pages in context
            debug: Enable debug logging
        """
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.context_window_size = context_window_size
        self.target_lang = target_lang
        self.bidirectional = bidirectional
        self.debug = debug

    def get_language_name(self, lang_code: str) -> str:
        """Convert language code to full language name."""
        language_map = {"ru": "Russian", "uk": "Ukrainian"}
        if lang_code not in language_map:
            raise ValueError(f"Unsupported language code: {lang_code}. Supported languages: ru, uk")
        return language_map[lang_code]

    def load_system_prompt(self, prompt_file_path: str) -> str:
        """Load system prompt from file and append language instruction."""
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                base_prompt = f.read().strip()

            # Add language instruction to the system prompt
            language_instruction = f"\n\nTranslate the content to {self.get_language_name(self.target_lang)}."
            return base_prompt + language_instruction

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

    def get_numbered_md_files(self, directory: str, target_pages: List[int] = None) -> List[Path]:
        """Get numbered markdown files, optionally filtered by target pages."""
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Find all numbered .md files
        md_files = []
        for file_path in directory_path.glob("*.md"):
            match = re.match(r'^(\d+)\.md$', file_path.name)
            if match:
                file_number = int(match.group(1))
                if not target_pages or file_number in target_pages:
                    md_files.append((file_number, file_path))

        if not md_files:
            raise ValueError(f"No numbered markdown files found in {directory}")

        # Sort by file number
        md_files.sort(key=lambda x: x[0])
        return [file_path for _, file_path in md_files]

    def read_markdown_file(self, file_path: Path) -> str:
        """Read content from markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {e}")

    def get_output_file_path(self, input_file_path: Path) -> Path:
        """Generate output file path for translated file."""
        match = re.match(r'^(\d+)\.md$', input_file_path.name)
        if not match:
            raise ValueError(f"Invalid input filename format: {input_file_path.name}")

        file_number = match.group(1)
        output_filename = f"{file_number}.{self.target_lang}.md"
        return input_file_path.parent / output_filename

    def load_context_if_needed(self, all_files: List[Path], current_file: Path):
        """This method is no longer needed as we use source files for context."""
        pass

    def create_context_content(self, all_files: List[Path], current_file: Path) -> str:
        """Create context content from previous and next source pages."""
        if self.context_window_size <= 0:
            return ""

        # Find current file position
        current_index = None
        for i, f in enumerate(all_files):
            if f == current_file:
                current_index = i
                break

        if current_index is None:
            return ""

        context_parts = []

        # Previous pages (source)
        start_index = max(0, current_index - self.context_window_size)
        for i in range(start_index, current_index):
            try:
                content = self.read_markdown_file(all_files[i])
                page_num = current_index - i
                context_parts.append(f"=== Previous Page -{page_num} (Source) ===\n{content}\n")
            except:
                break

        # Next pages (source) - only if bidirectional
        if self.bidirectional:
            next_end = min(len(all_files), current_index + self.context_window_size + 1)
            for i in range(current_index + 1, next_end):
                try:
                    content = self.read_markdown_file(all_files[i])
                    page_num = i - current_index
                    context_parts.append(f"=== Next Page +{page_num} (Source) ===\n{content}\n")
                except:
                    break

        return "\n".join(context_parts)

    def translate_page(self, content: str, system_prompt: str, file_name: str, all_files: List[Path], current_file: Path) -> str:
        """Translate a single page using OpenAI API with context."""
        context_content = self.create_context_content(all_files, current_file)

        user_message_parts = []
        if context_content:
            user_message_parts.append("CONTEXT (previous and next source pages for reference):")
            user_message_parts.append(context_content)
            user_message_parts.append("=" * 50)

        user_message_parts.append(f"TRANSLATE THE FOLLOWING CONTENT INTO {self.get_language_name(self.target_lang).upper()}:")
        user_message_parts.append(content)

        user_message = "\n".join(user_message_parts)

        try:
            print(f"Translating {file_name} to {self.get_language_name(self.target_lang)}...")

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
                temperature=0.3
            )

            translated_content = response.choices[0].message.content

            if self.debug:
                print("=" * 80)
                print("DEBUG: RESPONSE")
                print("=" * 80)
                print(translated_content)
                print("=" * 80 + "\n")

            return translated_content

        except Exception as e:
            raise Exception(f"Error translating {file_name}: {e}")

    def translate_directory(self, input_dir: str, system_prompt_file: str, page_range: str = None):
        """Translate numbered markdown files in directory."""
        system_prompt = self.load_system_prompt(system_prompt_file)
        target_pages = self.parse_pages(page_range) if page_range else None

        # Get all files first (for context loading)
        all_files = self.get_numbered_md_files(input_dir)
        # Get files to process
        md_files = self.get_numbered_md_files(input_dir, target_pages)

        print(f"Found {len(all_files)} total files, processing {len(md_files)} files")
        if target_pages:
            print(f"Target pages: {target_pages}")
        if self.bidirectional:
            print(f"Using bidirectional context: {self.context_window_size} pages before and after")

        try:
            for i, file_path in enumerate(md_files):
                print(f"\nProcessing file {i + 1}/{len(md_files)}: {file_path.name}")

                # Show context information
                current_index = None
                for idx, f in enumerate(all_files):
                    if f == file_path:
                        current_index = idx
                        break

                if current_index is not None and self.context_window_size > 0:
                    context_info = []

                    # Previous pages
                    start_index = max(0, current_index - self.context_window_size)
                    if start_index < current_index:
                        prev_pages = [all_files[j].stem for j in range(start_index, current_index)]
                        context_info.append(f"prev: {','.join(prev_pages)}")

                    # Next pages (if bidirectional)
                    if self.bidirectional:
                        next_end = min(len(all_files), current_index + self.context_window_size + 1)
                        if current_index + 1 < next_end:
                            next_pages = [all_files[j].stem for j in range(current_index + 1, next_end)]
                            context_info.append(f"next: {','.join(next_pages)}")

                    if context_info:
                        print(f"  Context: {' | '.join(context_info)}")

                output_path = self.get_output_file_path(file_path)

                content = self.read_markdown_file(file_path)
                translated_content = self.translate_page(content, system_prompt, file_path.name, all_files, file_path)

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(translated_content)

                print(f"  ✅ Saved translation: {output_path.name}")

            print(f"\n🎉 Translation completed successfully!")

        except Exception as e:
            print(f"❌ Error during translation: {e}")
            raise


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Translate numbered markdown files using OpenAI API with sliding context window"
    )
    parser.add_argument("input_directory", help="Directory containing numbered markdown files")
    parser.add_argument("--system-prompt-file", default="prompt.txt",
                        help="Path to system prompt file (default: prompt.txt)")
    parser.add_argument("--lang", default="ru", choices=["ru", "uk"],
                        help="Target language code: ru (Russian) or uk (Ukrainian) (default: ru)")
    parser.add_argument("--pages",
                        help="Page range to translate (examples: '5', '1,3,5', '1-5', '1-3,7,10-12')")
    parser.add_argument("--context-window", type=int, default=2,
                        help="Number of previous pages to include as context (default: 2)")
    parser.add_argument("--bidirectional", action="store_true",
                        help="Include both previous and next pages in context")
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
        translator = MarkdownTranslator(
            api_key=api_key,
            model=model,
            base_url=base_url,
            context_window_size=args.context_window,
            target_lang=args.lang,
            bidirectional=args.bidirectional,
            debug=args.debug
        )

        translator.translate_directory(
            args.input_directory,
            args.system_prompt_file,
            args.pages
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()