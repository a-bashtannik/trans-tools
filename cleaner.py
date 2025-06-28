import os
import re
import sys
import argparse

# Hardcoded fragments to remove (add your fragments here)
FRAGMENTS_TO_REMOVE = [
    "&</sup>lt;sup>",
]


def trim_page_content(content):
    """Remove trailing empty lines and --- separators inside <article> tags"""

    def replace_page_content(match):
        page_content = match.group(1)  # content inside <article> tags

        # Split into lines for processing
        lines = page_content.split('\n')

        # Remove trailing empty lines and lines with only ---
        while lines:
            last_line = lines[-1].strip()
            if last_line == '' or last_line == '---':
                lines.pop()
            else:
                break

        # Join back and remove any remaining trailing whitespace
        trimmed_content = '\n'.join(lines).rstrip()

        # Return formatted content with empty line after opening tag
        if trimmed_content:
            return f'<article>\n\n{trimmed_content}\n</article>'
        else:
            return '<article>\n\n</article>'

    # Match <article> tags with their content, capturing everything between them
    pattern = r'<article>\s*(.*?)\s*</article>'
    result = re.sub(pattern, replace_page_content, content, flags=re.DOTALL)
    return result


def process_markdown_files(folder_path="."):
    """Process all numbered markdown files in the specified folder"""

    print(f"Looking for .md files in: {folder_path}")

    # Get all files in the folder
    try:
        files = os.listdir(folder_path)
        print(f"Found {len(files)} files total")
    except Exception as e:
        print(f"Error reading folder: {e}")
        return

    # Filter and sort markdown files with digit names
    md_files = []
    for file in files:
        if re.match(r'^\d+\.md$', file):
            number = int(file.split('.')[0])
            md_files.append((number, file))
            print(f"Found markdown file: {file}")

    if not md_files:
        print("No numbered .md files found!")
        return

    # Sort by number
    md_files.sort(key=lambda x: x[0])
    print(f"Processing {len(md_files)} markdown files in order: {[f[1] for f in md_files]}")

    # Process each file
    for number, filename in md_files:
        filepath = os.path.join(folder_path, filename)
        print(f"\n--- Processing: {filename} ---")

        # Read file content
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"Read {len(content)} characters from {filename}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        original_content = content

        # Remove hardcoded fragments
        fragments_removed = 0
        for fragment in FRAGMENTS_TO_REMOVE:
            count_before = content.count(fragment)
            content = content.replace(fragment, '')
            fragments_removed += count_before
        if fragments_removed > 0:
            print(f"Removed {fragments_removed} hardcoded fragments")

        # Remove LaTeX math formulas
        latex_matches = re.findall(r'\$[^$]+\$', content)
        if latex_matches:
            print(f"Found {len(latex_matches)} LaTeX formulas: {latex_matches[:3]}{'...' if len(latex_matches) > 3 else ''}")
        content = re.sub(r'\$[^$]+\$', '', content)

        # Add horizontal line before first <sup> element that starts a line
        # and wrap content in appropriate tags
        lines = content.split('\n')
        sup_line_index = None

        # Find first <sup> that starts a line
        for i, line in enumerate(lines):
            if line.strip().startswith('<sup>'):
                sup_line_index = i
                print(f"Found <sup> at line {i}: {line.strip()[:50]}...")
                break

        if sup_line_index is not None:
            # Split content at <sup> line
            main_lines = lines[:sup_line_index]
            footer_lines = lines[sup_line_index:]

            main_content = '\n'.join(main_lines).strip()
            footer_content = '\n'.join(footer_lines).strip()

            print(f"Split content: main={len(main_content)} chars, footer={len(footer_content)} chars")

            # Wrap main content in <article> if not already wrapped
            if main_content and not main_content.startswith('<article>'):
                main_content = f'<article>\n\n{main_content}\n</article>'
                print("Wrapped main content in <article> tags")

            # Wrap footer content in <footer> if not already wrapped
            if footer_content and not footer_content.startswith('<footer>'):
                footer_content = f'<footer>\n{footer_content}\n</footer>'
                print("Wrapped footer content in <footer> tags")

            # Combine with separator
            if main_content and footer_content:
                content = f'{main_content}\n---\n{footer_content}'
                print("Combined main and footer with --- separator")
            elif main_content:
                content = main_content
                print("Only main content present")
            elif footer_content:
                content = footer_content
                print("Only footer content present")
        else:
            print("No <sup> found at start of line")
            # No <sup> found, wrap entire content in <article>
            content = content.strip()
            if content and not content.startswith('<article>'):
                content = f'<article>\n\n{content}\n</article>'
                print("Wrapped entire content in <article> tags")

        # Remove page number artifacts at the end of text
        lines = content.split('\n')
        if lines and (re.match(r'^\s*\d{1,3}\s*$', lines[-1]) or
                      re.match(r'^\s*\$\d{1,3}\$\s*$', lines[-1])):
            removed_line = lines[-1]
            lines.pop()
            print(f"Removed page number artifact: '{removed_line.strip()}'")
            # Remove any trailing empty lines
            empty_lines_removed = 0
            while lines and lines[-1].strip() == '':
                lines.pop()
                empty_lines_removed += 1
            if empty_lines_removed > 0:
                print(f"Removed {empty_lines_removed} trailing empty lines")
        content = '\n'.join(lines)

        # Trim empty lines inside <article> tags
        content = trim_page_content(content)
        print("Trimmed trailing empty lines inside <article> tags")

        # Write back to the same file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            if content != original_content:
                print(f"✓ Content changed, saved {len(content)} characters to {filename}")
            else:
                print(f"- No changes made to {filename}")

        except Exception as e:
            print(f"Error writing {filename}: {e}")

        print(f"Completed: {filename}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Clean and format markdown files by removing fragments and organizing content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cleaner.py ./output/
  python cleaner.py --folder ./output/
"""
    )

    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Directory containing markdown files to process (default: current directory)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    folder_path = args.folder

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist!")
        sys.exit(1)

    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory!")
        sys.exit(1)

    print(f"Processing files in folder: {folder_path}")
    process_markdown_files(folder_path)
    print("All files processed successfully!")