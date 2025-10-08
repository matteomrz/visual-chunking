import json
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import red, blue, green, orange, purple, black, transparent
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen.canvas import Canvas
import io

guideline_directory = "../guidelines"
annotated_directory = "../annotated/unstructured-io"
bounding_boxes_directory = "../bounding-boxes/unstructured-io"

# Color mapping for different element types
ELEMENT_COLORS = {
    "Title": red,
    "UncategorizedText": blue,
    "NarrativeText": green,
    "ListItem": orange,
    "Table": purple,
    "Image": black,
    "Footer": black,
    "Header": black,
}


def get_element_color(element_type):
    """Get color for element type, default to black if not found"""
    return ELEMENT_COLORS.get(element_type, black)


def draw_bboxes_on_pdf(input_pdf_path, output_pdf_path, json_data):
    """
    Draw bounding boxes on top of the original PDF
    """
    # Read the original PDF
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()

    # Process each page
    for page_num, page in enumerate(reader.pages):
        # Get page dimensions
        page_width = float(page.mediabox.width)
        page_height = float(page.mediabox.height)

        # Create overlay canvas
        packet = io.BytesIO()
        overlay_canvas = Canvas(packet, pagesize=(page_width, page_height))

        for element in json_data:
            metadata = element.get("metadata", {})
            coordinates = metadata.get("coordinates", {})
            element_page = metadata.get("page_number", 1)

            if element_page == page_num + 1:
                points = coordinates.get("points", [])
                element_type = element.get("type", "Unknown")
                layout_height = coordinates.get("layout_height", page_height)
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


def draw_bboxes(file_name):
    """
    Main function to draw bounding boxes on PDF
    """
    input_pdf_path = f"{guideline_directory}/{file_name}.pdf"
    output_pdf_path = f"{annotated_directory}/{file_name}-annotated.pdf"
    json_path = f"{bounding_boxes_directory}/{file_name}-output.json"

    # Check if files exist
    if not os.path.exists(input_pdf_path):
        print(f"Error: PDF file not found: {input_pdf_path}")
        return

    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return

    # Load JSON data
    with open(json_path, "r") as file:
        data = json.load(file)

    print(f"Processing {len(data)} elements...")

    # Draw bounding boxes
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
    draw_bboxes("example-guideline")
