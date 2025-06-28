import os
import re
import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam
)


def extract_article_content(file_path):
    """Extract content from <article> section"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    article_match = re.search(r'<article>(.*?)</article>', content, re.DOTALL)
    if article_match:
        return article_match.group(1)
    return None


def find_hyphen_files(directory):
    """Find files with hyphens in <article> section"""
    hyphen_files = []

    for file_path in sorted(Path(directory).glob('*.md')):
        article_content = extract_article_content(file_path)
        if article_content and article_content.rstrip().endswith('-'):
            hyphen_files.append(file_path)

    return hyphen_files


def get_next_files(current_file, directory, count=3):
    """Get next N files after current file"""
    current_num = int(current_file.stem)
    next_files = []

    for i in range(1, count + 1):
        next_file = Path(directory) / f"{current_num + i}.md"
        if next_file.exists():
            next_files.append(next_file)

    return next_files


def show_continuation_options(next_files, use_llm=False, original_phrase=""):
    """Show continuation options to user or let LLM decide"""
    print("\nSelect file with continuation:")

    options = []
    for i, file_path in enumerate(next_files, 1):
        article_content = extract_article_content(file_path)
        if article_content:
            preview = article_content.strip()[:100].replace('\n', ' ')
            print(f"{i}. {file_path.name}: {preview}...")
            options.append(f"{i}. {file_path.name}: {preview}...")
        else:
            print(f"{i}. {file_path.name}: (no <article> content)")
            options.append(f"{i}. {file_path.name}: (no <article> content)")

    if use_llm:
        return ask_llm_for_choice(original_phrase, options, next_files)

    while True:
        try:
            choice = int(input("\nEnter number (or 0 to skip): "))
            if choice == 0:
                return None
            if 1 <= choice <= len(next_files):
                return next_files[choice - 1]
            print("Invalid choice!")
        except ValueError:
            print("Enter a number!")


def ask_llm_for_choice(original_phrase, options, next_files):
    """Ask LLM to choose the correct continuation"""
    load_dotenv()

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise Exception("OPENAI_API_KEY not found in .env file")

    client = OpenAI(api_key=api_key)

    system_prompt = """You are helping to fix hyphenated word breaks in text. 
You will be given an original phrase ending with a hyphen and several continuation options.
Your task is to select which option contains the correct continuation of the hyphenated word.

Respond with ONLY the number (1, 2, or 3) of the correct option. Do not provide any explanation."""

    user_prompt = f"""Original phrase with hyphen: "{original_phrase}"

Options:
{chr(10).join(options)}

Which option contains the correct continuation? Respond with only the number."""

    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": system_prompt
    }
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": user_prompt
    }

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_message, user_message],
        temperature=0.1,
        max_tokens=10
    )

    llm_response = response.choices[0].message.content.strip()
    print(f"LLM selected: {llm_response}")

    choice = int(llm_response)
    if 1 <= choice <= len(next_files):
        print(f"✓ LLM chose option {choice}: {next_files[choice - 1].name}")
        return next_files[choice - 1]
    else:
        raise Exception(f"Invalid LLM choice: {choice}")


def manual_fallback_choice(next_files):
    """Manual fallback when LLM fails"""
    while True:
        try:
            choice = int(input("\nEnter number (or 0 to skip): "))
            if choice == 0:
                return None
            if 1 <= choice <= len(next_files):
                return next_files[choice - 1]
            print("Invalid choice!")
        except ValueError:
            print("Enter a number!")


def find_sentence_start(text, pos):
    """Find sentence start (after . ? !)"""
    sentence_ends = ['.', '?', '!']

    for i in range(pos - 1, -1, -1):
        if text[i] in sentence_ends:
            # Find start of next sentence (skip spaces)
            for j in range(i + 1, len(text)):
                if text[j].strip():
                    return j
            return i + 1

    return 0  # If not found, take from beginning


def process_hyphen_transfer(source_file, target_file):
    """Process sentence transfer between files"""
    # Read source file
    with open(source_file, 'r', encoding='utf-8') as f:
        source_content = f.read()

    # Read target file
    with open(target_file, 'r', encoding='utf-8') as f:
        target_content = f.read()

    # Extract <article> sections
    source_article = extract_article_content(source_file)
    target_article = extract_article_content(target_file)

    if not source_article or not target_article:
        print("Error: cannot find <article> sections")
        return False

    # Remove hyphen from end of source section
    source_article = source_article.rstrip()
    if source_article.endswith('-'):
        source_article = source_article[:-1]

    # Find sentence start in source file
    hyphen_pos = len(source_article)
    sentence_start = find_sentence_start(source_article, hyphen_pos)

    # Split source section
    remaining_source = source_article[:sentence_start].rstrip()
    sentence_part = source_article[sentence_start:]

    # Combine with beginning of target section
    target_article = target_article.lstrip()

    # Find first word in target file for combining
    first_word_match = re.match(r'(\S*)', target_article)
    if first_word_match:
        first_word = first_word_match.group(1)
        rest_target = target_article[len(first_word):]

        # Combine broken word
        combined_sentence = sentence_part + first_word + rest_target

        # Update file contents (preserve original formatting)
        new_source_content = re.sub(
            r'<article>(.*?)</article>',
            f'<article>{remaining_source}</article>',
            source_content,
            flags=re.DOTALL
        )

        new_target_content = re.sub(
            r'<article>(.*?)</article>',
            f'<article>{combined_sentence}</article>',
            target_content,
            flags=re.DOTALL
        )

        # Write changes
        with open(source_file, 'w', encoding='utf-8') as f:
            f.write(new_source_content)

        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(new_target_content)

        print(f"✓ Sentence moved from {source_file.name} to {target_file.name}")
        return True

    return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fix hyphenated word breaks in Polish text by moving content between files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hyphen_fix.py ./output/
  python hyphen_fix.py --use-llm ./output/
"""
    )

    parser.add_argument("directory", help="Directory containing numbered markdown files")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM to automatically select the correct continuation"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    directory = args.directory
    use_llm = args.use_llm

    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found!")
        sys.exit(1)

    # Find files with hyphens
    hyphen_files = find_hyphen_files(directory)

    if not hyphen_files:
        print("No files with hyphens found.")
        return

    print(f"Found {len(hyphen_files)} files with hyphens.")

    # Process each file
    for source_file in hyphen_files:
        print(f"\n{'=' * 50}")
        print(f"Processing file: {source_file.name}")

        # Show original phrase with hyphen
        source_article = extract_article_content(source_file)
        original_phrase = ""
        if source_article:
            # Find the sentence with hyphen and show some context
            lines = source_article.strip().split('\n')
            for line in reversed(lines):
                if line.strip().endswith('-'):
                    original_phrase = line.strip()
                    print(f"Original phrase: ...{original_phrase}")
                    break

        # Get next files
        next_files = get_next_files(source_file, directory)

        if not next_files:
            print("Next files not found. Skipping.")
            continue

        # Show options to user or LLM
        target_file = show_continuation_options(next_files, use_llm, original_phrase)

        if target_file:
            success = process_hyphen_transfer(source_file, target_file)
            if not success:
                print("Error processing transfer.")
        else:
            print("Skipped by user.")

    print("\nProcessing completed!")


if __name__ == "__main__":
    main()