#!/usr/bin/env python3
"""
Convert SOP Markdown to PDF with proper formatting
"""

import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
import os

def convert_md_to_pdf(md_file, pdf_file):
    """Convert markdown file to PDF with styling"""

    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=[
            'markdown.extensions.tables',
            'markdown.extensions.fenced_code',
            'markdown.extensions.codehilite',
            'markdown.extensions.toc',
        ]
    )

    # Create styled HTML document
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm;
                @top-right {{
                    content: "Page " counter(page) " of " counter(pages);
                    font-size: 9pt;
                    color: #666;
                }}
            }}

            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
                font-size: 10pt;
                line-height: 1.6;
                color: #333;
                max-width: 100%;
            }}

            h1 {{
                color: #2c3e50;
                font-size: 24pt;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-top: 30px;
                margin-bottom: 20px;
                page-break-after: avoid;
            }}

            h2 {{
                color: #34495e;
                font-size: 18pt;
                border-bottom: 2px solid #95a5a6;
                padding-bottom: 8px;
                margin-top: 25px;
                margin-bottom: 15px;
                page-break-after: avoid;
            }}

            h3 {{
                color: #34495e;
                font-size: 14pt;
                margin-top: 20px;
                margin-bottom: 12px;
                page-break-after: avoid;
            }}

            h4 {{
                color: #555;
                font-size: 12pt;
                margin-top: 15px;
                margin-bottom: 10px;
                page-break-after: avoid;
            }}

            code {{
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: "Courier New", Courier, monospace;
                font-size: 9pt;
                color: #c7254e;
            }}

            pre {{
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-left: 3px solid #3498db;
                padding: 12px;
                border-radius: 4px;
                overflow-x: auto;
                margin: 15px 0;
                page-break-inside: avoid;
            }}

            pre code {{
                background-color: transparent;
                padding: 0;
                color: #333;
                font-size: 9pt;
            }}

            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
                font-size: 9pt;
                page-break-inside: avoid;
            }}

            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}

            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}

            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}

            ul, ol {{
                margin: 10px 0;
                padding-left: 30px;
            }}

            li {{
                margin: 5px 0;
            }}

            blockquote {{
                border-left: 4px solid #3498db;
                padding-left: 15px;
                margin: 15px 0;
                color: #555;
                font-style: italic;
            }}

            strong {{
                color: #2c3e50;
                font-weight: 600;
            }}

            hr {{
                border: none;
                border-top: 2px solid #ecf0f1;
                margin: 30px 0;
            }}

            .page-break {{
                page-break-before: always;
            }}

            /* Prevent orphans and widows */
            p, li {{
                orphans: 3;
                widows: 3;
            }}

            /* Keep headers with content */
            h1, h2, h3, h4, h5, h6 {{
                page-break-after: avoid;
            }}

            /* Avoid breaking inside certain elements */
            table, pre, blockquote {{
                page-break-inside: avoid;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Convert HTML to PDF
    font_config = FontConfiguration()
    html = HTML(string=styled_html)
    html.write_pdf(
        pdf_file,
        font_config=font_config
    )

    print(f"✓ PDF created successfully: {pdf_file}")
    file_size = os.path.getsize(pdf_file) / 1024  # KB
    print(f"  File size: {file_size:.1f} KB")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_file = os.path.join(script_dir, "SOP_HARDWARE_REPLACEMENT.md")
    pdf_file = os.path.join(script_dir, "SOP_HARDWARE_REPLACEMENT.pdf")

    print("Converting SOP to PDF...")
    print(f"Input:  {md_file}")
    print(f"Output: {pdf_file}")
    print()

    convert_md_to_pdf(md_file, pdf_file)
