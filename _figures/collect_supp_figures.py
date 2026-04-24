from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.pdfgen import canvas as rl_canvas
from PIL import Image as PILImage
import io
from pathlib import Path
from pypdf import PdfReader, PdfWriter
import pikepdf

def create_supplementary_figures(
    images_data: list,  # List of dicts: {'path': str, 'title': str, 'legend': str}
    output_pdf: str,
    max_width_cm: float = 18,
    page_size=A4,
    dpi: float = 300,
    title: str = "Supplementary Figures",
):
    """
    Create supplementary figures PDF preserving vector graphics.
    
    Args:
        images_data: List of dicts with 'path', 'title', 'legend' keys
        output_pdf: Output PDF file path
        max_width_cm: Maximum image width in cm
        page_size: Page size for text pages
        dpi: DPI for raster images (default 300)
    """
    from pypdf import PdfWriter
    from io import BytesIO
    
    output_writer = PdfWriter()
    
    # Track page numbers for each figure (will be populated as we add pages)
    figure_page_numbers = []
    subfigure_page_numbers = {}

    # Build TOC entries (main figures + optional indented subfigures)
    toc_entries = _build_toc_entries(images_data)
    
    # Add TOC page and get entry positions
    entry_positions = _add_toc_page(output_writer, toc_entries, page_size)
    
    # Track page numbers for bookmarks
    current_page = 1  # TOC is page 0
    
    for idx, item in enumerate(images_data):
        # Get item-specific max_width_cm or use global default
        item_max_width = item.get('max_width_cm', max_width_cm)
        
        # Store this figure's starting page number
        figure_page_numbers.append(current_page)
        
        # Track pages before adding content
        pages_before = len(output_writer.pages)
        
        # Check if this is a combined figure (multiple subfigures)
        if 'paths' in item or 'subfigures' in item:
            pdf_paths = item.get('paths', item.get('subfigures', []))
            subfigure_page_numbers[idx] = [current_page + i for i in range(len(pdf_paths))]
            create_combined_figure_page(
                output_writer,
                pdf_paths,
                item.get('subtitles', []),
                item.get('title'),
                item.get('legend'),
                item_max_width,
                page_size,
                dpi
            )
        else:
            img_path = Path(item['path'])
            
            # Load image preserving vector format
            if img_path.suffix.lower() == '.svg':
                img_obj = _load_svg_as_vector(img_path, item_max_width)
                _add_page_with_content(
                    output_writer, img_obj, item.get('title'), item.get('legend'), page_size
                )
            elif img_path.suffix.lower() == '.pdf':
                # For PDF: add source page + create separate overlay with text
                _add_pdf_page_with_text_overlay(
                    output_writer, img_path, item.get('title'), item.get('legend'), 
                    item_max_width, page_size
                )
            else:  # PNG, JPG
                img_obj = _load_raster_image(img_path, item_max_width, dpi)
                _add_page_with_content(
                    output_writer, img_obj, item.get('title'), item.get('legend'), page_size
                )
        
        # Update current page counter
        pages_added = len(output_writer.pages) - pages_before
        current_page += pages_added
    
    # Add links to TOC page using the stored entry positions
    toc_target_pages = _get_toc_target_pages(toc_entries, figure_page_numbers, subfigure_page_numbers)
    _add_toc_links(output_writer, toc_target_pages, entry_positions)
    
    # Add hierarchical bookmarks
    parent_bookmark = output_writer.add_outline_item(title, 0)
    for idx, item in enumerate(images_data):
        title = item.get('title')
        if isinstance(title, tuple) and len(title) >= 2:
            bookmark_title = f"{title[0]}: {title[1]}"
        elif isinstance(title, tuple):
            bookmark_title = title[0]
        else:
            bookmark_title = title if title else f"Figure {idx+1}"
        
        figure_bookmark = output_writer.add_outline_item(bookmark_title, figure_page_numbers[idx], parent=parent_bookmark)

        if idx in subfigure_page_numbers:
            subtitles = item.get('subtitles', [])
            fig_num = title[0] if isinstance(title, tuple) and len(title) >= 1 else ""
            for sub_idx, page_num in enumerate(subfigure_page_numbers[idx]):
                if sub_idx < len(subtitles):
                    sup_entry = subtitles[sub_idx]
                    if isinstance(sup_entry, tuple):
                        label = sup_entry[0]
                        subtitle = sup_entry[1] if len(sup_entry) > 1 else None
                    else:
                        label = sup_entry
                        subtitle = None
                    sub_title = f"{fig_num}{label}"
                    if subtitle:
                        sub_title = f"{sub_title}: {subtitle}"
                else:
                    sub_title = f"{bookmark_title} - subplot {sub_idx + 1}"

                output_writer.add_outline_item(sub_title, page_num, parent=figure_bookmark)
    
    with open(output_pdf, 'wb') as f:
        output_writer.write(f)


def _build_toc_entries(images_data):
    """Build TOC entries including optional indented subfigure entries."""
    toc_entries = []
    for idx, item in enumerate(images_data):
        title = item.get('title')
        if isinstance(title, tuple) and len(title) >= 2:
            fig_num = title[0]
            description = title[1]
        elif isinstance(title, tuple):
            fig_num = title[0]
            description = None
        else:
            fig_num = title if title else f"Figure {idx+1}"
            description = None

        toc_entries.append({
            'level': 0,
            'type': 'main',
            'figure_idx': idx,
            'sub_idx': None,
            'fig_num': fig_num,
            'description': description,
        })

        if 'paths' in item or 'subfigures' in item:
            subtitles = item.get('subtitles', [])
            for sub_idx, sup_entry in enumerate(subtitles):
                if isinstance(sup_entry, tuple):
                    label = sup_entry[0]
                    subtitle = sup_entry[1] if len(sup_entry) > 1 else None
                else:
                    label = sup_entry
                    subtitle = None
                toc_entries.append({
                    'level': 1,
                    'type': 'sub',
                    'figure_idx': idx,
                    'sub_idx': sub_idx,
                    'fig_num': f"{fig_num}{label}",
                    'description': subtitle,
                })

    return toc_entries


def _get_toc_target_pages(toc_entries, figure_page_numbers, subfigure_page_numbers):
    """Resolve TOC entry targets to concrete page indices."""
    target_pages = []
    for entry in toc_entries:
        figure_idx = entry['figure_idx']
        if entry['type'] == 'main':
            target_pages.append(figure_page_numbers[figure_idx])
        else:
            sub_idx = entry['sub_idx']
            sub_pages = subfigure_page_numbers.get(figure_idx, [])
            if sub_idx is not None and sub_idx < len(sub_pages):
                target_pages.append(sub_pages[sub_idx])
            else:
                target_pages.append(figure_page_numbers[figure_idx])
    return target_pages


def _add_toc_page(writer, toc_entries, page_size):
    """Add a table of contents page listing all figures and return entry positions."""
    from io import BytesIO
    
    page_width_pts = float(page_size[0])
    page_height_pts = float(page_size[1])
    
    toc_buffer = BytesIO()
    c = rl_canvas.Canvas(toc_buffer, pagesize=page_size)
    
    # Title
    c.setFont("Times-Bold", 18)
    title_y = page_height_pts - 28.346 - 18  # top margin + font height
    c.drawCentredString(page_width_pts / 2, title_y, "Supplementary Figures")
    
    # Calculate starting position for entries
    # After title (18pt) + spaceAfter (20pt) + Spacer (0.5cm = 14.173pt)
    first_entry_y = title_y - 20 - 14.173
    
    # Entry font and spacing
    entry_font_size = 11
    entry_spacing = 8
    left_margin_pts = 28.346
    
    # Store entry positions for link creation
    entry_positions = []
    
    c.setFont("Times-Roman", entry_font_size)
    
    for idx, entry in enumerate(toc_entries):
        fig_num = entry.get('fig_num')
        description = entry.get('description')
        indent = 16 if entry.get('level') == 1 else 0
        entry_x = left_margin_pts + indent
        entry_y = first_entry_y - (idx * (entry_font_size + entry_spacing))

        c.setFont("Times-Bold", entry_font_size)
        c.drawString(entry_x, entry_y, fig_num)

        if description:
            fig_num_width = c.stringWidth(fig_num, "Times-Bold", entry_font_size)
            c.setFont("Times-Roman", entry_font_size)
            c.drawString(entry_x + fig_num_width, entry_y, f": {description}")
        
        # Store position for this entry (baseline is at y position)
        entry_positions.append({
            'y_baseline': entry_y,
            'left': entry_x,
            'right': page_width_pts - 28.346,
            'height': entry_font_size
        })
    
    c.save()
    toc_buffer.seek(0)
    
    toc_reader = PdfReader(toc_buffer)
    writer.add_page(toc_reader.pages[0])
    
    return entry_positions


def _add_toc_links(writer, target_pages, entry_positions):
    """Add clickable links to the TOC page using exact entry positions."""
    from pypdf.annotations import Link
    from pypdf.generic import Fit
    
    for idx, (page_num, pos) in enumerate(zip(target_pages, entry_positions)):
        # Use exact positions from text placement
        # Add small descent below baseline for click area
        entry_bottom = pos['y_baseline'] - 3
        entry_top = pos['y_baseline'] + pos['height']
        
        # Create link annotation using pypdf's Link class
        annotation = Link(
            rect=(pos['left'], entry_bottom, pos['right'], entry_top),
            target_page_index=page_num,
            fit=Fit(fit_type="/Fit")
        )
        
        writer.add_annotation(page_number=0, annotation=annotation)


def _add_page_with_content(writer, img_obj, title, legend, page_size):
    """Add a page with image, title, and legend using ReportLab."""
    from io import BytesIO
    
    doc_buffer = BytesIO()
    doc = SimpleDocTemplate(
        doc_buffer, pagesize=page_size,
        leftMargin=1*cm, rightMargin=1*cm,
        topMargin=1*cm, bottomMargin=1*cm
    )
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=12,
        leading=14,
        spaceAfter=12,
    )
    legend_style = ParagraphStyle(
        'CustomLegend',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10,
        leading=16,
        alignment=TA_JUSTIFY,
        hyphenationLang='en_US',
        hyphenationMinWordLength=6,
        spaceAfter=6
    )
    
    story = []
    if title:
        # Title is tuple: (fig_num, description)
        if isinstance(title, tuple) and len(title) >= 2:
            full_title = f"<b>{title[0]}</b>: {title[1]}"
        elif isinstance(title, tuple):
            full_title = f"<b>{title[0]}</b>"
        else:
            full_title = title
        story.append(Paragraph(full_title, title_style))
        story.append(Spacer(1, 0.3*cm))
    
    story.append(img_obj)
    story.append(Spacer(1, 0.5*cm))
    
    if legend:
        story.append(Paragraph(legend, legend_style))
    
    doc.build(story)
    doc_buffer.seek(0)
    
    # Add this page to output
    from pypdf import PdfReader
    reader = PdfReader(doc_buffer)
    writer.add_page(reader.pages[0])


def _add_pdf_page_with_text_overlay(writer, pdf_path, title, legend, max_width_cm, page_size):
    """Add a PDF page as vector with figure number in header, plus title/legend text page."""
    from pypdf import PdfReader, Transformation
    from io import BytesIO
    
    page_width_cm = float(page_size[0]) / 28.346
    full_page_width_pts = page_width_cm * 28.346
    header_height_pts = 40  # Space for label
    
    # Read source PDF and scale to max_width_cm (may be less than page width)
    pdf_reader = PdfReader(str(pdf_path))
    first_page = pdf_reader.pages[0]
    
    mediabox = first_page.mediabox
    page_width_pts = float(mediabox.width)
    page_height_pts = float(mediabox.height)
    
    max_width_pts = max_width_cm * 28.346
    if page_width_pts > max_width_pts:
        scale = max_width_pts / page_width_pts
        first_page.scale(scale, scale)
        page_width_pts = max_width_pts
        page_height_pts = page_height_pts * scale
    
    # Create new page with header space (full page width)
    total_height_pts = page_height_pts + header_height_pts
    combined_buffer = BytesIO()
    c = rl_canvas.Canvas(combined_buffer, pagesize=(full_page_width_pts, total_height_pts))
    
    # Draw white background only for header area
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, page_height_pts, full_page_width_pts, header_height_pts, fill=1, stroke=0)
    
    # Add figure number in header if title is provided
    if title:
        fig_num = title[0] if isinstance(title, tuple) else title
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Times-Bold", 12)
        c.drawString(28.346, total_height_pts - 25, fig_num)  # 1cm margin to match title/legend
    
    c.save()
    combined_buffer.seek(0)
    
    # Merge: place the figure left-aligned at the bottom
    combined_reader = PdfReader(combined_buffer)
    combined_page = combined_reader.pages[0]
    
    # Merge the figure page (left-aligned)
    combined_page.merge_page(first_page)
    
    # Add combined page
    writer.add_page(combined_page)
    
    # Create text page with title and legend
    if title or legend:
        estimated_height_cm = 8
        custom_page_size = (full_page_width_pts, estimated_height_cm * 28.346)
        
        text_buffer = BytesIO()
        text_doc = SimpleDocTemplate(
            text_buffer, pagesize=custom_page_size,
            leftMargin=1*cm, rightMargin=1*cm,
            topMargin=1*cm, bottomMargin=1*cm
        )
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Normal'],
            fontName='Times-Roman',
            fontSize=14,
            leading=16,
            spaceAfter=12,
        )
        legend_style = ParagraphStyle(
            'CustomLegend',
            parent=styles['Normal'],
            fontName='Times-Roman',
            fontSize=10,
            leading=16,
            alignment=TA_JUSTIFY,
            hyphenationLang='en_US',
            hyphenationMinWordLength=6,
            spaceAfter=6
        )
        
        story = []
        if title:
            # Title is tuple: (fig_num, description)
            if isinstance(title, tuple) and len(title) >= 2:
                full_title = f"<b>{title[0]}</b>: {title[1]}"
            elif isinstance(title, tuple):
                full_title = f"<b>{title[0]}</b>"
            else:
                full_title = title
            story.append(Paragraph(full_title, title_style))
        if legend:
            story.append(Paragraph(legend, legend_style))
        
        text_doc.build(story)
        text_buffer.seek(0)
        
        text_reader = PdfReader(text_buffer)
        text_page = text_reader.pages[0]
        text_mediabox = text_page.mediabox
        actual_height_pts = float(text_mediabox.height)
        
        if actual_height_pts < estimated_height_cm * 28.346:
            text_page.mediabox = (0, 0, full_page_width_pts, actual_height_pts)
        
        writer.add_page(text_page)

def _load_svg_as_vector(svg_path: Path, max_width_cm: float):
    """Load SVG as vector drawing (stays vector in PDF)."""
    drawing = svg2rlg(str(svg_path))
    
    # Scale to max width preserving aspect ratio
    if drawing.width > max_width_cm * 28.346:  # cm to points
        scale = (max_width_cm * 28.346) / drawing.width
        drawing.scale(scale, scale)
    
    return drawing

def _load_pdf_page_as_vector(pdf_path: Path, max_width_cm: float):
    """Load PDF first page as vector content by creating overlay with title/legend."""
    from pypdf import PdfReader, PdfWriter
    from reportlab.pdfgen import canvas as rl_canvas
    from io import BytesIO
    
    # Read source PDF first page
    reader = PdfReader(str(pdf_path))
    first_page = reader.pages[0]
    
    # Get original page dimensions
    mediabox = first_page.mediabox
    page_width_pts = float(mediabox.width)
    page_height_pts = float(mediabox.height)
    
    # Calculate scaling
    max_width_pts = max_width_cm * 28.346
    scale = min(1.0, max_width_pts / page_width_pts)
    
    scaled_width_pts = page_width_pts * scale
    scaled_height_pts = page_height_pts * scale
    
    # Convert to cm for reportlab
    width_cm = scaled_width_pts / 28.346
    height_cm = scaled_height_pts / 28.346
    
    # Create wrapper object that will embed as vector in final PDF
    # Return a special marker object
    return {
        'type': 'pdf_vector',
        'path': pdf_path,
        'width_cm': width_cm,
        'height_cm': height_cm
    }

def _load_raster_image(img_path: Path, max_width_cm: float, dpi: float = 300):
    """Load raster image (PNG, JPG) with scaling.
    
    Args:
        img_path: Path to image file
        max_width_cm: Maximum width in cm
        dpi: DPI for the output (default 300)
    """
    pil_img = PILImage.open(img_path)
    orig_width, orig_height = pil_img.size
    
    # Convert max_width from cm to pixels at target DPI
    max_width_px = max_width_cm * dpi / 2.54  # 2.54 cm per inch
    
    # Scale to max width if needed
    if orig_width > max_width_px:
        scale = max_width_px / orig_width
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        pil_img = pil_img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
    
    # Calculate size in cm based on DPI
    width_cm = (pil_img.width * 2.54) / dpi
    height_cm = (pil_img.height * 2.54) / dpi
    
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return RLImage(img_bytes, width=width_cm*cm, height=height_cm*cm)


def create_combined_figure_page(writer, pdf_paths, subtitles, title, legend, max_width_cm, page_size, dpi=300):
    """
    Create pages with multiple figures (PDF/PNG/SVG), each as a separate page with label.
    Adds subfigures first, then title page, then legend page.
    
    Args:
        writer: PdfWriter to add pages to
        pdf_paths: List of Path objects to figure files (PDF, PNG, SVG, JPG)
        subtitles: List of tuples (label, subtitle) for each subfigure
        title: Tuple of (fig_num, description) e.g. ('Fig. S1', 'Description')
        legend: Legend text for the combined figure
        max_width_cm: Maximum width for figures
        page_size: Page size for text pages
        dpi: DPI for raster images (default 300)
    """
    from io import BytesIO
    
    page_width_cm = float(page_size[0]) / 28.346
    full_page_width_pts = page_width_cm * 28.346
    styles = getSampleStyleSheet()
    
    # Get figure number for subfigure labels
    fig_num = title[0] if isinstance(title, tuple) and len(title) >= 1 else ""
    header_height_pts = 40  # Space for label
    max_width_pts = max_width_cm * 28.346
    
    # Add each subfigure as its own page with label in header
    for idx, fig_path in enumerate(pdf_paths):
        fig_path = Path(fig_path)
        
        # Convert to PDF page first if not already PDF
        if fig_path.suffix.lower() == '.pdf':
            pdf_reader = PdfReader(str(fig_path))
            page = pdf_reader.pages[0]
            
            # Scale to max_width (may be less than page width)
            mediabox = page.mediabox
            width_pts = float(mediabox.width)
            height_pts = float(mediabox.height)
            
            if width_pts > max_width_pts:
                scale = max_width_pts / width_pts
                page.scale(scale, scale)
                width_pts = max_width_pts
                height_pts = height_pts * scale
        else:
            # For PNG/SVG: create a temporary PDF page fitted exactly to image size
            from reportlab.graphics import renderPDF
            from reportlab.lib.utils import ImageReader

            temp_buffer = BytesIO()

            if fig_path.suffix.lower() == '.svg':
                drawing = _load_svg_as_vector(fig_path, max_width_cm)
                img_width_pts = drawing.width
                img_height_pts = drawing.height
                temp_canvas = rl_canvas.Canvas(temp_buffer, pagesize=(img_width_pts, img_height_pts))
                renderPDF.draw(drawing, temp_canvas, 0, 0)
                temp_canvas.save()
            else:  # PNG, JPG
                pil_img = PILImage.open(fig_path)
                orig_width, orig_height = pil_img.size

                max_width_px = max_width_cm * dpi / 2.54
                if orig_width > max_width_px:
                    scale = max_width_px / orig_width
                    new_width = int(orig_width * scale)
                    new_height = int(orig_height * scale)
                    pil_img = pil_img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)

                img_width_pts = (pil_img.width / dpi) * 72.0
                img_height_pts = (pil_img.height / dpi) * 72.0

                img_bytes = io.BytesIO()
                pil_img.save(img_bytes, format='PNG')
                img_bytes.seek(0)

                temp_canvas = rl_canvas.Canvas(temp_buffer, pagesize=(img_width_pts, img_height_pts))
                temp_canvas.drawImage(
                    ImageReader(img_bytes),
                    0,
                    0,
                    width=img_width_pts,
                    height=img_height_pts,
                    preserveAspectRatio=True,
                    mask='auto'
                )
                temp_canvas.save()

            temp_buffer.seek(0)
            temp_reader = PdfReader(temp_buffer)
            page = temp_reader.pages[0]

            mediabox = page.mediabox
            width_pts = float(mediabox.width)
            height_pts = float(mediabox.height)
        
        # Create new page with header space (full page width)
        total_height_pts = height_pts + header_height_pts
        combined_buffer = BytesIO()
        c = rl_canvas.Canvas(combined_buffer, pagesize=(full_page_width_pts, total_height_pts))
        
        # Draw white background only for header area
        c.setFillColorRGB(1, 1, 1)
        c.rect(0, height_pts, full_page_width_pts, header_height_pts, fill=1, stroke=0)
        
        # Add label in header (e.g., "Fig. S1A: This is a subtitle")
        if subtitles and idx < len(subtitles):
            entry = subtitles[idx]
            if isinstance(entry, tuple):
                label = entry[0]
                subtitle = entry[1] if len(entry) > 1 else None
            else:
                label = entry
                subtitle = None

            subfig_label = f"{fig_num}{label}"
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Times-Bold", 12)
            x_pos = 28.346
            y_pos = total_height_pts - 25
            c.drawString(x_pos, y_pos, subfig_label)

            if subtitle:
                bold_width = c.stringWidth(subfig_label, "Times-Bold", 12)
                c.setFont("Times-Roman", 12)
                c.drawString(x_pos + bold_width, y_pos, f": {subtitle}")
        
        c.save()
        combined_buffer.seek(0)
        
        # Merge: place the figure left-aligned at the bottom
        combined_reader = PdfReader(combined_buffer)
        combined_page = combined_reader.pages[0]
        
        # Merge the figure page (left-aligned)
        combined_page.merge_page(page)
        
        writer.add_page(combined_page)
    
    # Add title and legend on one page after subfigures
    if title or legend:
        estimated_height_cm = 8
        custom_page_size = (full_page_width_pts, estimated_height_cm * 28.346)
        
        text_buffer = BytesIO()
        text_doc = SimpleDocTemplate(
            text_buffer, pagesize=custom_page_size,
            leftMargin=1*cm, rightMargin=1*cm,
            topMargin=1*cm, bottomMargin=1*cm
        )
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Normal'],
            fontName='Times-Roman',
            fontSize=12,
            leading=14,
            spaceAfter=12,
        )
        legend_style = ParagraphStyle(
            'CustomLegend',
            parent=styles['Normal'],
            fontName='Times-Roman',
            fontSize=10,
            leading=16,
            alignment=TA_JUSTIFY,
            hyphenationLang='en_US',
            hyphenationMinWordLength=6,
            spaceAfter=6
        )
        
        story = []
        if title:
            # Format as "Fig. S1: Description"
            if isinstance(title, tuple) and len(title) >= 2:
                full_title = f"<b>{title[0]}</b>: {title[1]}"
            elif isinstance(title, tuple):
                full_title = f"<b>{title[0]}</b>"
            else:
                full_title = title
            story.append(Paragraph(full_title, title_style))
        
        if legend:
            story.append(Paragraph(legend, legend_style))
        
        text_doc.build(story)
        text_buffer.seek(0)
        
        text_reader = PdfReader(text_buffer)
        text_page = text_reader.pages[0]
        text_mediabox = text_page.mediabox
        actual_height_pts = float(text_mediabox.height)
        
        if actual_height_pts < estimated_height_cm * 28.346:
            text_page.mediabox = (0, 0, full_page_width_pts, actual_height_pts)
        
        writer.add_page(text_page)


# -----------------------

wd = Path(__file__).parent.parent
output_pdf = wd / "_figures" /"supplementary_figures.pdf"
images_data = [
    {
        "title": ("Fig. S1", "NEOFC curves derived from resting-state network probability"),
        "path": wd / "plots" / "evaluate" / "posctrl_parc-Schaefer200_measure-pearson.pdf",
        "legend": """
            NEOFC curves for all 12 RSN probability atlases used as positive controls (Schaefer200 parcellation).
            Each panel shows mean rsFC as a function of the density percentile threshold for the original atlas (blue) 
            and its spatially inverted reference (dark red).
            Colored shaded area: 90% PI across individuals; grey shaded area: null distribution.
            Right inset: individual AUC+ and AUC&#8722; scores (each dot: one participant). 
            Displayed as in <b>Fig. 2b</b>.
            Brain maps indicate the parcellated RSN probability atlas, converted to percentiles.
            Abbreviations: rsFC: resting-state functional connectivity; AUC: area under the curve; RSN: resting-state network; PI: percentile interval.
        """
    },
    {
        "title": ("Fig. S2", "NEOFC curves derived from neurobiological atlases"),
        "path": wd / "plots" / "discover" / f"mri_parc-Schaefer200_measure-pearson_run-1.pdf",
        "legend": """
            NEOFC curves for all 25 nuclear imaging reference maps, organized by broad neurotransmitter systems (Schaefer200 and Schaefer200 + subcortex parcellation).
            Each panel shows mean rsFC as a function of the density percentile threshold for the original atlas (colored by neurotransmitter system) 
            and its spatially inverted reference (dark red).
            Colored shaded area: 90% PI across individuals; grey shaded area: null distribution.
            Brain maps indicate the parcellated reference atlas, converted to percentiles.
            Statistics printed in each plot include the raw p values for AUC+ and AUC&#8722, the ICC(2,k), and the WCV for AUC+.
            Displayed as in <b>Fig. 2d</b>.
            Abbreviations: rsFC: resting-state functional connectivity; PI: percentile interval; ICC: intraclass correlation coefficient; 
            WCV: within-subject coefficient of variability; see <b>Fig. 2</b> for reference map abbreviations.
        """
    },
    {
        "title": ("Fig. S3", "Individual-level p values associated to AUC scores"),
        "subtitles": [("a", "Schaefer200"), ("b", "Schaefer200+Subcortical")],
        "paths": [
            wd / "plots" / "discover" / f"mrisubjectsig_parc-{parc}_measure-pearson.pdf"
            for parc in ["Schaefer200", "Schaefer200Subcortical"]
        ],
        "max_width_cm": 14,
        "legend": """
            Distribution of individual-level &#8722;log10(p) values for AUC+ (left) and AUC&#8722; (right) across 25 nuclear imaging reference maps.
            <b>a</b>: Schaefer200 parcellation; <b>b</b>: Schaefer200 + subcortex parcellation.
            Each dot represents one participant; color indicates &#8722;log10(p).
            Vertical lines indicate significance thresholds at p &lt; 0.05 (dotted), p &lt; 0.01 (dashed), and p &lt; 0.001 (dot-dashed).
            Abbreviations: AUC: area under the curve; see <b>Fig. 2</b> for reference map abbreviations.
        """
    },
    {
        "title": ("Fig. S4", "Replication of AUC profiles in adult control samples"),
        "path": wd / "plots" / "replicability" / "auc_parc-Schaefer200.pdf",
        "legend": """
            AUC+ and AUC&#8722; scores (z-scored against a null distribution of spatially matched reference maps) for 25 nuclear imaging reference maps, 
            shown separately for six independent adult control samples.
            Displayed as in <b>Fig. 2c</b> (Schaefer200 and Schaefer200 + subcortex parcellations).
            Abbreviations: AUC: area under the curve; PI: percentile interval; WCV: within-subject coefficient of variability; 
            see <b>Fig. 2</b> for reference map abbreviations.
        """
    },
    {
        "title": ("Fig. S5", "Meta-Analysis of AUC p values from six MRI datasets"),
        "path": wd / "plots" / "replicability" / "metap_parc-Schaefer200.pdf",
        "max_width_cm": 10,
        "legend": """
            Meta-analytic Z scores (weighted Stouffer&#8217;s method) for AUC+ and AUC&#8722; across 25 nuclear imaging reference maps, 
            combining raw p values from six independent fMRI cohorts weighted by sample size.
            Darker bars: Schaefer200; brighter bars: Schaefer200 + subcortex; significance markers indicate pMeff.
            Abbreviations: AUC: area under the curve; pMeff: p-value corrected for effective number of comparisons; see <b>Fig. 2</b> for reference map abbreviations.
        """
    },
    {
        "title": ("Fig. S6", "AUC results including AUC-delta for different parcellation resolutions"),
        "subtitles": [("a", "Schaefer100(+Subcortical)"), ("b", "Schaefer200(+Subcortical)"), ("c", "Schaefer400(+Subcortical)")],
        "paths": [
            wd / "plots" / "discover" / f"mrioverview_parc-{parc}_measure-pearson.pdf"
            for parc in ["Schaefer100", "Schaefer200", "Schaefer400"]
        ],
        "legend": """
            AUC+, AUC&#8722;, and AUC&#916; (AUC+ minus AUC&#8722;) scores for 25 nuclear imaging reference maps,
            shown for three parcellation resolutions: <b>a</b> Schaefer100 (+subcortex), <b>b</b> Schaefer200 (+subcortex), <b>c</b> Schaefer400 (+subcortex).
            Displayed as in <b>Fig. 2c</b>.
            Abbreviations: AUC: area under the curve; PI: percentile interval; ICC: intraclass correlation coefficient; 
            see <b>Fig. 2</b> for reference map abbreviations.
        """
    },
    {
        "title": ("Fig. S7", "AUC results without interhemispheric connections and an alternative aggregate metric"),
        "subtitles": [("a", "No interhemispheric connections"), ("b", "Polynomial fit aggregation")],
        "paths": [
            wd / "plots" / "discover" / f"mrioverview_parc-Schaefer200_measure-pearson_nointernodelta.pdf",
            wd / "plots" / "discover" / f"mrioverview_parc-Schaefer200_measure-pearson_stat-poly2_nodelta.pdf"
        ],
        "legend": """
            AUC+ and AUC&#8722; scores for 25 nuclear imaging reference maps under two methodological variants (Schaefer200 parcellation).
            <b>a</b> Replication of the main analysis excluding interhemispheric connections.
            <b>b</b> Replication using a 2nd-degree polynomial fit coefficient (Poly2) as an alternative to AUC as the aggregate metric.
            Displayed as in <b>Fig. 2c</b>.
            Abbreviations: AUC: area under the curve; Poly2: 2nd-degree polynomial fit coefficient; PI: percentile interval;
            ICC: intraclass correlation coefficient; see <b>Fig. 2</b> for reference map abbreviations.
        """
    },
    {
        "title": ("Fig. S8", "NEOFC curves across MEG frequency bands"),
        "subtitles": [("a", "delta"), ("b", "theta"), ("c", "alpha"), ("d", "beta"), ("e", "lgamma"), ("f", "hgamma")],
        "paths": [
            wd / "plots" / "discover" / f"meg_parc-Schaefer200_measure-aec_fq-{fqband}.pdf"
            for fqband in ["delta", "theta", "alpha", "beta", "lgamma", "hgamma"]
        ],
        "legend": """
            NEOFC curves for all 25 nuclear imaging reference maps derived from MEG AEC FC matrices,
            shown for six frequency bands: <b>a</b> delta, <b>b</b> theta, <b>c</b> alpha, <b>d</b> beta, <b>e</b> low gamma, <b>f</b> high gamma.
            Displayed as in <b>Fig. S2</b> (Schaefer200 parcellation).
            Abbreviations: AEC: amplitude envelope correlation; FC: functional connectivity; PI: percentile interval;
            see <b>Fig. 2</b> for reference map abbreviations.
        """
    },
    {
        "title": ("Fig. S9", "Correspondence of fMRI and MEG AUC profiles across all frequency bands"),
        "path": wd / "plots" / "comp_mri_meg" / "indivprofile_parc-Schaefer200.pdf",
        "legend": """
            Correspondence between fMRI and MEG AUC profiles (z-scored against null) across 25 reference maps,
            shown for all six MEG frequency bands (rows) and four AUC combinations (columns):
            fMRI AUC+ vs. MEG AUC+, fMRI AUC&#8722; vs. MEG AUC&#8722;, fMRI AUC+ vs. MEG AUC&#8722;, fMRI AUC&#8722; vs. MEG AUC+.
            Displayed as in <b>Fig. 3c</b> (Schaefer200 parcellation).
            Abbreviations: AUC: area under the curve; fMRI: functional magnetic resonance imaging;
            MEG: magnetoencephalography; CI: confidence interval; see <b>Fig. 2</b> for reference map abbreviations.
        """
    },
    {
        "title": ("Fig. S10", "MEG AUC results with different parcellation resolutions"),
        "subtitles": [("a", "Schaefer100"), ("b", "Schaefer400")],
        "paths": [
            wd / "plots" / "discover" / f"megoverview_parc-Schaefer100_measure-aec.pdf",
            wd / "plots" / "discover" / f"megoverview_parc-Schaefer400_measure-aec.pdf",
        ],
        "max_width_cm": 15,
        "legend": """
            AUC+ and AUC&#8722; scores for 25 nuclear imaging reference maps derived from MEG AEC FC matrices,
            shown for two alternative parcellation resolutions: <b>a</b> Schaefer100, <b>b</b> Schaefer400.
            Displayed as in <b>Fig. 3b</b>.
            Abbreviations: AUC: area under the curve; AEC: amplitude envelope correlation; FC: functional connectivity;
            see <b>Fig. 2</b> for reference map abbreviations.
        """
    },
    {
        "title": ("Fig. S11", "Correlation of atlas-wise AUC scores between MRI and MEG"),
        "path": wd / "plots" / "comp_mri_meg" / "subjectcorr_parc-Schaefer200.pdf",
        "max_width_cm": 15,
        "legend": """
            Spearman&#8217;s rho between individual fMRI and MEG AUC profiles for 25 nuclear imaging reference maps
            and six MEG frequency bands, shown for AUC+ (left) and AUC&#8722; (right) (Schaefer200 parcellation).
            Color: Spearman&#8217;s rho; markers indicate p &lt; 0.05.
            Abbreviations: AUC: area under the curve; fMRI: functional magnetic resonance imaging;
            MEG: magnetoencephalography; see <b>Fig. 2</b> for reference map abbreviations.
        """
    },
    {
        "title": ("Fig. S12", "Leave-one-out analysis to identify influential regions and connections"),
        "subtitles": [("a", "AUC+"), ("b", "AUC–")],
        "paths": [
            wd / "plots" / "loo" / f"regionalimportance_metric-{metric}_parc-Schaefer200Subcortical.png"
            for metric in ["original", "inverted"]
        ],
        "legend": """
            Regional (left, leave-one-region-out) and connection (right, leave-one-connection-out) influence on AUC
            for all 25 nuclear imaging reference maps (Schaefer200 + subcortex parcellation).
            <b>a</b> AUC+ maps; <b>b</b> AUC&#8722; maps.
            Color indicates direction and magnitude of influence.
            Displayed as in <b>Fig. 4a</b>.
            Abbreviations: AUC: area under the curve; see <b>Fig. 2</b> for reference map abbreviations.
        """
    },
    {
        "title": ("Fig. S13", "NEOFC curves with AUC estimation restricted to lowered percentiles"),
        "subtitles": [("a", "Schaefer200"), ("b", "Schaefer200+Subcortical")],
        "paths": [
            wd / "plots" / "discover" / f"mriaucthresh_parc-{parc}_run-1.pdf"
            for parc in ["Schaefer200", "Schaefer200Subcortical"]
        ],
        "legend": """
            Sensitivity analysis assessing whether AUC effects are driven by the highest-density atlas regions.
            The maximum percentile threshold included in AUC calculation was systematically varied from 5 to 90,
            and significance reported as a function of this threshold for all 25 nuclear imaging reference maps.
            <b>a</b> Schaefer200 parcellation; <b>b</b> Schaefer200 + subcortex parcellation.
            Abbreviations: AUC: area under the curve; pMeff: p-value corrected for effective number of comparisons;
            see <b>Fig. 2</b> for reference map abbreviations.
        """
    },
    {
        "title": ("Fig. S14", "Effects of covariate regression on NET AUC results"),
        "path": wd / "plots" / "covariates" / f"covoverview_parc-Schaefer200_measure-pearson.pdf",
        "max_width_cm": 15,
        "legend": """
            Sensitivity analysis testing whether the NET AUC+ effect is driven by spatially confounding factors.
            Each spatial covariate map was individually regressed from the NET atlas and NEOFC AUC+ estimation was rerun on the residuals (Schaefer200 parcellation).
            Covariate maps included T1/T2 ratio, sensory-association axis, BigBrain gradient maps,
            gray matter and cerebrospinal fluid probability maps, and probability maps of cerebral veins and arteries.
            AUC+ results after regression are shown in comparison to the unregressed baseline.
            Displayed as in <b>Fig. 2c</b>.
            Abbreviations: AUC: area under the curve; PI: percentile interval; 
            NET: norepinephrine/noradrenaline transporter; SA: sensory-association; GM: gray matter; CSF: cerebrospinal fluid.
        """
    },
    {
        "title": ("Fig. S15", "Quality control of HCP-YA physiological data (heart-rate variability)"),
        "path": wd / "plots" / "physio" / f"hrv_qc.pdf",
        "max_width_cm": 15,
        "legend": """
            Quality control of HRV estimates derived from PPG recordings (HCP-YA dataset).
            Left: empirical cumulative distribution of mean PPG signal quality index; 
            red dashed line indicates the data-driven knee-point threshold used for session exclusion.
            Center and right: RMSSD and MadNN as a function of mean PPG quality index; 
            color indicates included (green) and excluded (red) sessions based on quality threshold and upper HRV bound (150 ms).
            Abbreviations: HRV: heart rate variability; PPG: photoplethysmography; RMSSD: root mean square of successive differences; 
            MadNN: median absolute deviation of NN intervals.
        """
    },
    {
        "title": ("Fig. S16", "Quality control of YRSP physiological data (pupil diameter over time)"),
        "path": wd / "plots" / "physio" / f"pupil_qc.pdf",
        "legend": """
            Raw (red) and cleaned (green) pupil diameter time series for both runs of each participant in the YRSP dataset.
            Panel titles indicate the proportion of valid data points after artifact removal.
            Sessions with fewer than 80% valid data points were excluded from analyses.
            Abbreviations: YRSP: Yale Resting-State Pupillometry dataset.
        """
    },
    {
        "title": ("Fig. S17", "Associations of AUC scores with clinical variables"),
        "path": wd / "plots" / "clinic" / f"clinicassoc_parc-Schaefer200.pdf",
        "legend": """
            Associations of NEOFC AUC+ and AUC&#8722; scores (Schaefer200 and Schaefer200 + subcortex parcellations) 
            with six clinical and cognitive outcomes in the psychosis group.
            Both Spearman and partial Spearman correlations were computed, adjusting for sex, age, mean FD, global connectivity, and current antipsychotic dose.
            Correlations with antipsychotic outcomes were calculated without the antipsychotic covariate.
            Abbreviations: AUC: area under the curve; PANSS: Positive and Negative Syndrome Scale; CPZ: chlorpromazine;
            FD: framewise displacement; Meff: effective number of tests.
        """
    },

]

create_supplementary_figures(
    images_data=images_data,
    output_pdf=output_pdf,
    max_width_cm=20,
    page_size=A4,
    dpi=400
)


