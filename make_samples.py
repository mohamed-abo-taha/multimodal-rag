"""Generate sample multi-modal documents for testing the ingestion pipeline.

Creates:
  sample_docs/invoice.pdf  — a real text-layer PDF (tests pypdf extraction)
  sample_docs/poster.png   — an image containing text (tests Groq vision OCR)

Run once:  .venv/Scripts/python.exe make_samples.py
Requires:  pip install -r requirements-dev.txt  (fpdf2; Pillow comes with the app)
"""
import os

from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF
from fpdf.enums import XPos, YPos

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sample_docs")
os.makedirs(OUT, exist_ok=True)


def make_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 14, "INVOICE", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=12)
    lines = [
        "Invoice number: INV-2026-0042",
        "Date: June 15, 2026",
        "Bill to: Globex Industries",
        " ",
        "Line items:",
        "  - Annual SaaS subscription      $12,000",
        "  - Onboarding & training          $2,500",
        "  - Priority support add-on        $1,800",
        " ",
        "Subtotal: $16,300",
        "Tax (8%): $1,304",
        "Total due: $17,604",
        " ",
        "Payment terms: Net 30",
        "Due date: July 15, 2026",
    ]
    for ln in lines:
        pdf.cell(0, 8, ln, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    path = os.path.join(OUT, "invoice.pdf")
    with open(path, "wb") as f:
        f.write(pdf.output())
    return path


def make_image():
    width, height = 1000, 700
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font_path = r"C:\Windows\Fonts\arial.ttf"
        title_font = ImageFont.truetype(font_path, 46)
        body_font = ImageFont.truetype(font_path, 32)
    except OSError:
        title_font = body_font = ImageFont.load_default()

    draw.rectangle([0, 0, width, 120], fill=(20, 80, 160))
    draw.text((40, 35), "Annual Accessibility Summit 2026", font=title_font, fill="white")

    lines = [
        "Date: September 12-13, 2026",
        "Venue: Berlin Congress Center",
        "Keynote: Designing for WCAG 2.2",
        "Workshops: Automated vs. manual testing",
        "Register at: summit.example.org",
    ]
    y = 200
    for ln in lines:
        draw.text((50, y), ln, font=body_font, fill=(20, 20, 20))
        y += 75
    path = os.path.join(OUT, "poster.png")
    img.save(path)
    return path


if __name__ == "__main__":
    print("Created:", make_pdf())
    print("Created:", make_image())
