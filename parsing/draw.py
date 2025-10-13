import argparse
import io
import json
import os

from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.colors import black, blue, green, orange, purple, red, yellow, grey
from reportlab.pdfgen.canvas import Canvas

from config import ANNOTATED_DIR, BOUNDING_BOX_DIR, DEFAULT_GUIDELINE, DEFAULT_MODULE, GUIDELINES_DIR

parser = argparse.ArgumentParser()

parser.add_argument("--draw", "-d", action="store_true", help="Create annotated PDF")
parser.add_argument(
    "--file",
    "-f",
    type=str,
    default=DEFAULT_GUIDELINE,
    help=f'PDF filename without extension. Default: "{DEFAULT_GUIDELINE}"',
)
parser.add_argument(
    "--module",
    "-m",
    type=str,
    default=DEFAULT_MODULE,
    help=f'The output of which module to draw. Default: "{DEFAULT_MODULE}"',
)
parser.add_argument(
    "--appendix",
    "-a",
    type=str,
    default="",
    help=f'Optional: Any appendix attached to the path of the input JSON and output PDF. Example: "guideline\'-no_ocr\'.json"',
)

args = parser.parse_args()

# Color mapping for different element types
ELEMENT_COLORS = {
    # Unstructured.io Tags
    "Title": red,
    "UncategorizedText": blue,
    "NarrativeText": green,
    "ListItem": orange,
    "Table": purple,
    "Image": purple,
    "Footer": grey,
    "Header": grey,

    # LlamaParse Tags
    "picture": purple,
    "text": blue,
    "sectionHeader": red,
    "pageFooter": grey,
    "pageHeader": grey,
    "listItem": orange,
}


def get_element_color(element_type):
    """Get color for element type, default to black if not found"""
    if element_type not in ELEMENT_COLORS:
        print(f'Error: Element type "{element_type}" not found in ELEMENT_COLORS. Defaulting to black')
    return ELEMENT_COLORS.get(element_type, black)


def draw_bboxes_on_pdf(input_pdf_path, output_pdf_path, json_data):
    """
    Draw bounding boxes on top of the original PDF
    """
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()

    for page_num, page in enumerate(reader.pages):
        page_width = float(page.mediabox.width)
        page_height = float(page.mediabox.height)

        packet = io.BytesIO()
        overlay_canvas = Canvas(packet, pagesize=(page_width, page_height))

        for element in json_data:
            metadata = element.get("metadata", {})
            layout = metadata.get("layout", {})
            element_page = layout.get("page_num", 1)

            if element_page == page_num + 1:
                points = layout.get("bbox", [])
                element_type = metadata.get("type", "Unknown")
                layout_height = layout.get("height", page_height)
                scale_factor = layout_height / page_height

                if len(points) >= 4:
                    x1, y1 = [p / scale_factor for p in points[0]]  # top-left
                    x3, y3 = [p / scale_factor for p in points[2]]  # bottom-right

                    pdf_y1 = page_height - y1
                    pdf_y3 = page_height - y3

                    color = get_element_color(element_type)

                    # Draw rectangle
                    overlay_canvas.setStrokeColor(color)
                    overlay_canvas.setLineWidth(2)

                    # Draw the bounding box
                    overlay_canvas.rect(
                        x1, pdf_y1, x3 - x1, pdf_y3 - pdf_y1, stroke=1, fill=0
                    )

                    # Add Type Label
                    overlay_canvas.setFillColor(color)
                    overlay_canvas.setFont("Helvetica", 8)
                    overlay_canvas.drawString(x1, pdf_y1 + 5, f"{element_type}")

        overlay_canvas.save()
        packet.seek(0)

        overlay_pdf = PdfReader(packet)

        if overlay_pdf.pages:
            page.merge_page(overlay_pdf.pages[0])
            writer.add_page(page)

    # Write Output PDF
    with open(output_pdf_path, "wb") as output_file:
        writer.write(output_file)


def draw_bboxes(file_name, module_name, appendix=""):
    """
    Main function to draw bounding boxes on PDF
    """
    input_pdf_path = GUIDELINES_DIR / f"{file_name}.pdf"
    output_folder_path = ANNOTATED_DIR / module_name
    if appendix:
        output_pdf_path = output_folder_path / f"{file_name}-{appendix}-annotated.pdf"
        json_path = BOUNDING_BOX_DIR / module_name / f"{file_name}-{appendix}-output.json"
    else:
        output_pdf_path = output_folder_path / f"{file_name}-annotated.pdf"
        json_path = BOUNDING_BOX_DIR / module_name / f"{file_name}-output.json"

    # Check if files exist
    if not input_pdf_path.exists():
        print(f"Error: PDF file not found: {input_pdf_path}")
        return

    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return

    # Load JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"Processing {len(data)} elements...")

    # Draw bounding boxes
    os.makedirs(output_folder_path, exist_ok=True)
    draw_bboxes_on_pdf(input_pdf_path, output_pdf_path, data)
    print(f"Annotated PDF saved to: {output_pdf_path}")

    # Print summary
    element_types = {}
    for element in data:
        element_type = element.get("type", "Unknown")
        element_types[element_type] = element_types.get(element_type, 0) + 1

    print("\nElement type summary:")
    for element_type, count in element_types.items():
        print(f"  {element_type}: {count} elements")


if __name__ == "__main__":
    draw_bboxes(args.file, args.module, args.appendix)
