"""File management for Virtual Agora multi-file reports.

This module provides functionality to organize, save, and manage
multi-file report outputs with proper structure and naming.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ReportFileManager:
    """Manage multi-file report generation and organization."""

    def __init__(self, base_output_dir: Path = Path("outputs/reports")):
        """Initialize ReportFileManager.

        Args:
            base_output_dir: Base directory for report outputs.
        """
        self.base_output_dir = base_output_dir
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.current_report_dir = None
        self.file_manifest = []

    def create_report_directory(
        self, session_id: str, main_topic: Optional[str] = None
    ) -> Path:
        """Create a dedicated directory for a report.

        Args:
            session_id: Unique session identifier.
            main_topic: Main discussion topic for naming.

        Returns:
            Path to the created report directory.
        """
        # Create directory name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if main_topic:
            # Create safe directory name from topic
            safe_topic = re.sub(r"[^\w\s-]", "", main_topic)[:30]
            safe_topic = re.sub(r"[-\s]+", "_", safe_topic)
            dir_name = f"report_{timestamp}_{safe_topic}"
        else:
            dir_name = f"report_{timestamp}_{session_id[:8]}"

        # Create the directory
        self.current_report_dir = self.base_output_dir / dir_name
        self.current_report_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created report directory: {self.current_report_dir}")
        return self.current_report_dir

    def save_report_section(
        self, section_number: int, section_title: str, content: str
    ) -> Path:
        """Save a report section to a numbered file.

        Args:
            section_number: Section number for ordering.
            section_title: Title of the section.
            content: Section content in Markdown.

        Returns:
            Path to the saved file.
        """
        if not self.current_report_dir:
            raise ValueError("No report directory created yet")

        # Create filename
        safe_title = re.sub(r"[^\w\s-]", "", section_title)
        safe_title = re.sub(r"[-\s]+", "_", safe_title)
        filename = f"{section_number:02d}_{safe_title}.md"

        # Handle subsections (remove leading spaces and dashes)
        if section_title.strip().startswith("-"):
            # This is a subsection, adjust numbering
            filename = f"{section_number:02d}_sub_{safe_title}.md"

        file_path = self.current_report_dir / filename

        # Save the content
        file_path.write_text(content, encoding="utf-8")

        # Update manifest
        self.file_manifest.append(
            {
                "section_number": section_number,
                "section_title": section_title,
                "filename": filename,
                "file_path": str(file_path.relative_to(self.base_output_dir)),
                "size_bytes": file_path.stat().st_size,
            }
        )

        logger.info(f"Saved section '{section_title}' to {filename}")
        return file_path

    def create_table_of_contents(self, report_structure: List[str]) -> Path:
        """Create a table of contents file.

        Args:
            report_structure: Ordered list of section titles.

        Returns:
            Path to the table of contents file.
        """
        if not self.current_report_dir:
            raise ValueError("No report directory created yet")

        lines = [
            "# Table of Contents",
            "",
            f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Report Sections",
            "",
        ]

        # Add links to each section
        for i, section in enumerate(report_structure, 1):
            # Get the corresponding filename from manifest
            manifest_entry = next(
                (m for m in self.file_manifest if m["section_title"] == section), None
            )

            if manifest_entry:
                filename = manifest_entry["filename"]
                if section.strip().startswith("-"):
                    # Subsection with indentation
                    lines.append(f"  - [{section.strip('- ')}](./{filename})")
                else:
                    # Main section
                    lines.append(f"{i}. [{section}](./{filename})")
            else:
                # Section not yet written
                if section.strip().startswith("-"):
                    lines.append(f"  - {section.strip('- ')} *(pending)*")
                else:
                    lines.append(f"{i}. {section} *(pending)*")

        # Add file listing
        lines.extend(
            [
                "",
                "## File Listing",
                "",
                "| Section | File | Size |",
                "|---------|------|------|",
            ]
        )

        for entry in self.file_manifest:
            size_kb = entry["size_bytes"] / 1024
            lines.append(
                f"| {entry['section_title']} | "
                f"{entry['filename']} | "
                f"{size_kb:.1f} KB |"
            )

        # Save table of contents
        toc_path = self.current_report_dir / "00_Table_of_Contents.md"
        toc_path.write_text("\n".join(lines), encoding="utf-8")

        logger.info("Created table of contents")
        return toc_path

    def create_metadata_file(self, metadata: Dict[str, Any]) -> Path:
        """Create a metadata file for the report.

        Args:
            metadata: Report metadata dictionary.

        Returns:
            Path to the metadata file.
        """
        if not self.current_report_dir:
            raise ValueError("No report directory created yet")

        # Enhance metadata with file information
        metadata["report_info"] = {
            "generated_at": datetime.now().isoformat(),
            "report_directory": str(self.current_report_dir.name),
            "total_sections": len(self.file_manifest),
            "total_size_kb": sum(f["size_bytes"] for f in self.file_manifest) / 1024,
            "files": self.file_manifest,
        }

        # Save as JSON
        metadata_path = self.current_report_dir / "report_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

        # Also create a human-readable version
        readme_lines = [
            "# Report Information",
            "",
            f"**Generated**: {metadata['report_info']['generated_at']}",
            f"**Directory**: {metadata['report_info']['report_directory']}",
            f"**Total Sections**: {metadata['report_info']['total_sections']}",
            f"**Total Size**: {metadata['report_info']['total_size_kb']:.1f} KB",
            "",
        ]

        if "session_id" in metadata:
            readme_lines.append(f"**Session ID**: {metadata['session_id']}")
        if "duration_minutes" in metadata:
            readme_lines.append(f"**Duration**: {metadata['duration_minutes']} minutes")
        if "topics_discussed" in metadata:
            readme_lines.extend(
                [
                    "",
                    "## Topics Discussed",
                    "",
                ]
            )
            for topic in metadata["topics_discussed"]:
                readme_lines.append(f"- {topic}")

        readme_path = self.current_report_dir / "README.md"
        readme_path.write_text("\n".join(readme_lines), encoding="utf-8")

        logger.info("Created metadata files")
        return metadata_path

    def create_report_manifest(self) -> Path:
        """Create a manifest file listing all report components.

        Returns:
            Path to the manifest file.
        """
        if not self.current_report_dir:
            raise ValueError("No report directory created yet")

        manifest = {
            "manifest_version": "1.0",
            "created_at": datetime.now().isoformat(),
            "report_directory": str(self.current_report_dir.name),
            "files": [],
        }

        # List all files in the directory
        for file_path in sorted(self.current_report_dir.iterdir()):
            if file_path.is_file():
                manifest["files"].append(
                    {
                        "filename": file_path.name,
                        "size_bytes": file_path.stat().st_size,
                        "modified_at": datetime.fromtimestamp(
                            file_path.stat().st_mtime
                        ).isoformat(),
                    }
                )

        # Save manifest
        manifest_path = self.current_report_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        logger.info("Created report manifest")
        return manifest_path

    def validate_report_structure(self) -> Dict[str, Any]:
        """Validate the report structure and completeness.

        Returns:
            Validation results dictionary.
        """
        if not self.current_report_dir:
            return {"valid": False, "error": "No report directory"}

        validation = {
            "valid": True,
            "directory": str(self.current_report_dir),
            "issues": [],
            "file_count": 0,
            "total_size_kb": 0,
        }

        # Check for required files
        required_files = [
            "00_Table_of_Contents.md",
            "README.md",
            "report_metadata.json",
            "manifest.json",
        ]

        for req_file in required_files:
            file_path = self.current_report_dir / req_file
            if not file_path.exists():
                validation["valid"] = False
                validation["issues"].append(f"Missing required file: {req_file}")

        # Count and measure all files
        for file_path in self.current_report_dir.iterdir():
            if file_path.is_file():
                validation["file_count"] += 1
                validation["total_size_kb"] += file_path.stat().st_size / 1024

        # Check for section files
        section_files = [
            f
            for f in self.current_report_dir.glob("*.md")
            if f.name not in ["00_Table_of_Contents.md", "README.md"]
        ]

        if not section_files:
            validation["valid"] = False
            validation["issues"].append("No section files found")

        validation["section_count"] = len(section_files)

        return validation

    def get_report_path(self) -> Optional[Path]:
        """Get the current report directory path.

        Returns:
            Path to the report directory or None.
        """
        return self.current_report_dir

    def archive_report(self, archive_format: str = "zip") -> Path:
        """Create an archive of the report directory.

        Args:
            archive_format: Archive format (zip, tar, gztar).

        Returns:
            Path to the created archive.
        """
        if not self.current_report_dir:
            raise ValueError("No report directory to archive")

        # Create archive
        archive_path = shutil.make_archive(
            base_name=str(self.current_report_dir),
            format=archive_format,
            root_dir=self.current_report_dir.parent,
            base_dir=self.current_report_dir.name,
        )

        logger.info(f"Created report archive: {archive_path}")
        return Path(archive_path)
