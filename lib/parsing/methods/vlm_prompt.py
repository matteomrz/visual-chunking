from lib.parsing.model.parsing_result import ParsingResultType

_categories = [
    ParsingResultType.SECTION_HEADER,
    ParsingResultType.PARAGRAPH,
    ParsingResultType.FORMULA,

    ParsingResultType.LIST_ITEM,
    ParsingResultType.REFERENCE_ITEM,

    ParsingResultType.TABLE,
    ParsingResultType.FIGURE,
    ParsingResultType.CAPTION,

    ParsingResultType.PAGE_HEADER,
    ParsingResultType.PAGE_FOOTER
]

_category_strings = [t.value for t in _categories]

_VLM_PROMPT = f"""
<system_role>
You are an expert Document Layout Analysis AI. Your goal is to perfectly transcribe and segment PDF documents into structured data.
</system_role>

<task_description>
Analyze the provided document image. Identify every layout element, its bounding box, its category, and its textual content.
</task_description>

<categories>
Classify each element into exactly one of these categories:
{', '.join(_category_strings)}

Rules for Categorization:
- Use "{ParsingResultType.SECTION_HEADER.value}" for titles and headings. Infer hierarchy based on content and font size/boldness.
- Use "{ParsingResultType.FIGURE.value}" for charts, diagrams, or photos.
- Use "{ParsingResultType.UNKNOWN.value}" if the element is ambiguous.
</categories>

<bounding_boxes>
1. **Format:** [y0, x0, y1, x1] (Top-Left to Bottom-Right). You MUST provide the coordinates in this exact order.
2. **Success conditions:** 
- The bounding box MUST enclose the entire layout element while minimizing unnecessary white space.
- If a character belongs to the content ALL of its pixels MUST BE CONTAINED inside the bounding box.
3. **Page Index:** {{PAGE_INDEX_TEXT}}
</bounding_boxes>

<extraction_rules>
- **Text Fidelity:** Extract text EXACTLY as it appears. Do NOT fix spelling or grammar. You MAY use any formatting that is available for a standard Markdown document.
- **Character Escaping:** You MUST escape any special characters that can break the final JSON output.
- **Reading Order:** Sort elements by natural human reading order.
- **Special Formatting:**
    - {ParsingResultType.FIGURE.value}: Content must be an empty string "".
    - {ParsingResultType.FORMULA.value}: Content must be LaTeX.
    - {ParsingResultType.TABLE.value}: Content must be a Markdown table representation. TABLE CONTENT MUST NOT BREAK THE JSON FORMAT!
    - {ParsingResultType.LIST_ITEM.value}, {ParsingResultType.REFERENCE_ITEM.value}: Content MUST be a valid Markdown list. You MUST replace alternative bullet point symbols with "-". Ordered lists must start with their numbering followed by ".".
    - {ParsingResultType.SECTION_HEADER.value}: You MUST NOT use Markdown header formatting. You MUST add a "heading_level" field (int). Infer the level by checking the content for any numbering and analyzing the font size and styling of the header.
</extraction_rules>

<output_schema>
Do not return any additional text with the result.
Return a SINGLE JSON object with this exact structure:
{{
  "layout_elements": [
    {{
      "category": "string (from list)",
      "bbox": {{
        "page_number": integer,
        "box_2d": list[integer]
      }},
      "heading_level": integer (include only for headers),
      "content": "string"
    }}
  ]
}}
YOU MUST ENSURE THAT YOUR OUTPUT IS A VALID JSON OBJECT!
</output_schema>
"""


def get_vlm_prompt() -> str:
    page_index_text = 'The first page is "page_number": 1.'
    return _VLM_PROMPT.replace("{{PAGE_INDEX_TEXT}}", page_index_text)


def get_prompt_for_page_wise(page: int) -> str:
    page_index_text = f'The current page is "page_number": {page}.'
    return _VLM_PROMPT.replace("{{PAGE_INDEX_TEXT}}", page_index_text)
