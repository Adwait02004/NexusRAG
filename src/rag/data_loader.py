import os
import json
from pathlib import Path
from typing import List, Any
import pandas as pd
from langchain_core.documents import Document

def load_single_document(file_path: str | Path) -> List[Document]:
    """
    Robust universal multi-format document loader with granular metadata.
    Supported formats:
    - PDF (.pdf) via PyMuPDF / PyPDF
    - Word (.docx, .doc) via python-docx & docx2txt
    - Excel (.xlsx, .xls) via openpyxl & pandas (multi-sheet support)
    - JSON / JSONL (.json, .jsonl) via native standard library (zero jq dependencies)
    - CSV / TSV (.csv, .tsv) via pandas
    - PowerPoint (.pptx) via python-pptx (slide-by-slide tracking)
    - Text & Code (.txt, .md, .py, .yaml, .yml, .xml, .log, .sql) via resilient text decoders
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    docs: List[Document] = []

    try:
        # 1. PDF Documents
        if ext == ".pdf":
            try:
                import fitz  # PyMuPDF
                doc_pdf = fitz.open(str(path))
                for page_idx, page in enumerate(doc_pdf, start=1):
                    text = page.get_text().strip()
                    if text:
                        docs.append(Document(
                            page_content=text,
                            metadata={"source": str(path), "source_name": path.name, "page": page_idx, "file_type": "pdf"}
                        ))
                doc_pdf.close()
            except Exception as e_pdf:
                # Fallback to PyPDFLoader
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(str(path))
                for page_idx, doc in enumerate(loader.load(), start=1):
                    doc.metadata["page"] = page_idx
                    doc.metadata["source_name"] = path.name
                    doc.metadata["file_type"] = "pdf"
                    docs.append(doc)

        # 2. Word Documents (.docx, .doc)
        elif ext in [".docx", ".doc"]:
            try:
                import docx
                d_file = docx.Document(str(path))
                text_blocks = []
                for p in d_file.paragraphs:
                    if p.text.strip():
                        text_blocks.append(p.text.strip())
                for table in d_file.tables:
                    for row in table.rows:
                        row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                        if row_text:
                            text_blocks.append(row_text)
                full_text = "\n\n".join(text_blocks)
                if full_text.strip():
                    docs.append(Document(
                        page_content=full_text,
                        metadata={"source": str(path), "source_name": path.name, "page": 1, "file_type": "docx"}
                    ))
            except Exception:
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(str(path))
                for doc in loader.load():
                    doc.metadata["page"] = 1
                    doc.metadata["source_name"] = path.name
                    doc.metadata["file_type"] = "docx"
                    docs.append(doc)

        # 3. Excel Spreadsheets (.xlsx, .xls)
        elif ext in [".xlsx", ".xls"]:
            excel_file = pd.ExcelFile(str(path))
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                df = df.dropna(how="all")
                if df.empty:
                    continue
                # Format tabular data clearly as markdown
                try:
                    table_md = df.to_markdown(index=False)
                except Exception:
                    table_md = df.to_string(index=False)
                sheet_content = f"# Excel Sheet: {sheet_name}\n\n**File:** {path.name}\n\n{table_md}"
                docs.append(Document(
                    page_content=sheet_content,
                    metadata={"source": str(path), "source_name": path.name, "sheet": sheet_name, "page": 1, "file_type": "excel"}
                ))

        # 4. JSON & JSON Lines (.json, .jsonl)
        elif ext in [".json", ".jsonl"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                if ext == ".jsonl":
                    lines = [json.loads(line) for line in f if line.strip()]
                    formatted_json = json.dumps(lines, indent=2, ensure_ascii=False)
                else:
                    data = json.load(f)
                    formatted_json = json.dumps(data, indent=2, ensure_ascii=False)
            
            docs.append(Document(
                page_content=f"# JSON Dataset: {path.name}\n\n```json\n{formatted_json}\n```",
                metadata={"source": str(path), "source_name": path.name, "page": 1, "file_type": "json"}
            ))

        # 5. CSV & TSV (.csv, .tsv)
        elif ext in [".csv", ".tsv"]:
            sep = "\t" if ext == ".tsv" else ","
            try:
                df = pd.read_csv(str(path), sep=sep)
                try:
                    table_md = df.to_markdown(index=False)
                except Exception:
                    table_md = df.to_string(index=False)
                content = f"# CSV Table: {path.name}\n\n{table_md}"
                docs.append(Document(
                    page_content=content,
                    metadata={"source": str(path), "source_name": path.name, "page": 1, "file_type": "csv"}
                ))
            except Exception:
                from langchain_community.document_loaders import CSVLoader
                loader = CSVLoader(str(path), encoding="utf-8")
                for i, doc in enumerate(loader.load(), start=1):
                    doc.metadata["page"] = 1
                    doc.metadata["row"] = i
                    doc.metadata["source_name"] = path.name
                    doc.metadata["file_type"] = "csv"
                    docs.append(doc)

        # 6. PowerPoint Presentations (.pptx)
        elif ext == ".pptx":
            from pptx import Presentation
            prs = Presentation(str(path))
            for slide_num, slide in enumerate(prs.slides, start=1):
                slide_lines = []
                for shape in slide.shapes:
                    if shape.has_text_frame and shape.text_frame.text.strip():
                        slide_lines.append(shape.text_frame.text.strip())
                if slide_lines:
                    slide_content = f"# Slide {slide_num}: {path.name}\n\n" + "\n\n".join(slide_lines)
                    docs.append(Document(
                        page_content=slide_content,
                        metadata={"source": str(path), "source_name": path.name, "page": slide_num, "file_type": "pptx"}
                    ))

        # 7. Images (.png, .jpg, .jpeg, .webp)
        elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
            from PIL import Image
            img = Image.open(str(path))
            width, height = img.size
            format_name = img.format or ext.lstrip('.').upper()
            mode = img.mode
            img_info = f"# Image Asset: {path.name}\n\n**Format:** {format_name}\n**Resolution:** {width}x{height} px\n**Color Mode:** {mode}\n"
            
            exif_data = getattr(img, "_getexif", lambda: None)()
            if exif_data:
                img_info += f"**Metadata Tags:** {len(exif_data)} EXIF attributes present.\n"
            
            docs.append(Document(
                page_content=img_info,
                metadata={"source": str(path), "source_name": path.name, "page": 1, "file_type": "image"}
            ))

        # 8. Text, Markdown, and Source Code (.txt, .md, .py, .yaml, .xml, .log)
        else:
            raw_text = ""
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    raw_text = path.read_text(encoding=enc)
                    break
                except Exception:
                    continue
            if raw_text.strip():
                docs.append(Document(
                    page_content=raw_text,
                    metadata={"source": str(path), "source_name": path.name, "page": 1, "file_type": ext.lstrip(".")}
                ))

    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        raise RuntimeError(f"Error parsing '{path.name}': {e}")

    return docs


def load_all_documents(data_dir: str | Path) -> List[Document]:
    """Scan directory recursively and load all supported documents."""
    data_path = Path(data_dir).resolve()
    documents: List[Document] = []

    if not data_path.exists():
        return documents

    supported_exts = {
        ".pdf", ".txt", ".csv", ".docx", ".doc", ".xlsx",
        ".xls", ".json", ".jsonl", ".pptx", ".md", ".tsv"
    }

    for file_path in data_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_exts:
            # Skip hidden files or database files
            if file_path.name.startswith(".") or "vector_store" in str(file_path):
                continue
            try:
                loaded = load_single_document(file_path)
                documents.extend(loaded)
            except Exception as e:
                print(f"[WARN] Skipping {file_path.name}: {e}")

    print(f"[INFO] Loaded {len(documents)} document sections from {data_path}")
    return documents