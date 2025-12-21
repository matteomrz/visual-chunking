from pathlib import Path

import pymupdf


def pdf_to_page_img_bytes(pdf_path: Path, img_extension: str = "png"):
    doc = pymupdf.open(pdf_path)

    page_images = []
    for page in doc.pages():
        if isinstance(page, pymupdf.Page):
            page_bytes = page.get_pixmap().tobytes(output=img_extension)
            page_images.append(page_bytes)

    return page_images
