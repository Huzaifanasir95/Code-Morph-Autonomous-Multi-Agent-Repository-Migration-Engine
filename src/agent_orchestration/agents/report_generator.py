"""
Report Generator Agent

Generates comprehensive migration reports.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List

from rich.console import Console
from rich.table import Table

from src.agent_orchestration.schemas.orchestration_models import (
    FileInfo,
    MigrationReport,
    MigrationStatus,
    OrchestrationState,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Generates migration reports"""

    def __init__(self):
        """Initialize report generator"""
        self.console = Console()

    def generate_report(self, state: OrchestrationState) -> MigrationReport:
        """
        Generate final migration report

        Args:
            state: Orchestration state

        Returns:
            MigrationReport
        """
        logger.info("Generating migration report")

        # Collect all files
        all_files: List[FileInfo] = []
        for batch in state.batches:
            all_files.extend(batch.files)

        # Count by status
        successful = sum(
            1
            for f in all_files
            if f.status
            in [MigrationStatus.COMPLETED, MigrationStatus.VERIFIED]
        )
        failed = sum(1 for f in all_files if f.status == MigrationStatus.FAILED)
        verified = sum(1 for f in all_files if f.status == MigrationStatus.VERIFIED)
        verification_failed = sum(
            1 for f in all_files if f.status == MigrationStatus.VERIFICATION_FAILED
        )

        # Calculate rates
        total = len(all_files)
        success_rate = (successful / total * 100) if total > 0 else 0.0
        verification_rate = (verified / total * 100) if total > 0 else 0.0

        report = MigrationReport(
            repository=state.repository_path,
            target_framework=state.target_framework,
            total_files=total,
            successful=successful,
            failed=failed,
            verified=verified,
            verification_failed=verification_failed,
            success_rate=success_rate,
            verification_rate=verification_rate,
            total_duration_seconds=state.total_duration_ms / 1000,
            files=all_files,
            errors=state.errors,
            timestamp=datetime.now().isoformat(),
        )

        logger.info(
            f"Report generated: {successful}/{total} successful, "
            f"{verified} verified, {failed} failed"
        )

        return report

    def print_report(self, report: MigrationReport) -> None:
        """
        Print report to console

        Args:
            report: Migration report
        """
        self.console.print("\n[bold]Migration Report[/bold]\n")

        # Summary table
        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")

        summary_table.add_row("Repository", report.repository)
        summary_table.add_row("Target Framework", report.target_framework)
        summary_table.add_row("Total Files", str(report.total_files))
        summary_table.add_row(
            "Successful", f"{report.successful} ({report.success_rate:.1f}%)"
        )
        summary_table.add_row(
            "Verified", f"{report.verified} ({report.verification_rate:.1f}%)"
        )
        summary_table.add_row("Failed", str(report.failed))
        summary_table.add_row("Verification Failed", str(report.verification_failed))
        summary_table.add_row(
            "Duration", f"{report.total_duration_seconds:.1f}s"
        )

        self.console.print(summary_table)

        # File details
        if report.files:
            self.console.print("\n[bold]File Details:[/bold]\n")

            files_table = Table(show_header=True, header_style="bold magenta")
            files_table.add_column("File", style="cyan")
            files_table.add_column("Status", style="white")
            files_table.add_column("Complexity", style="yellow")
            files_table.add_column("Verification", style="green")

            for file in report.files[:20]:  # Show first 20
                status_color = self._get_status_color(file.status)
                verification = (
                    f"{file.verification_score:.1%}"
                    if file.verification_score is not None
                    else "N/A"
                )

                files_table.add_row(
                    file.path,
                    f"[{status_color}]{file.status.value}[/{status_color}]",
                    file.complexity,
                    verification,
                )

            if len(report.files) > 20:
                files_table.add_row(
                    f"... and {len(report.files) - 20} more",
                    "",
                    "",
                    "",
                )

            self.console.print(files_table)

        # Errors
        if report.errors:
            self.console.print(f"\n[bold red]Errors ({len(report.errors)}):[/bold red]")
            for error in report.errors[:10]:
                self.console.print(f"  • {error}")

        # Overall status
        if report.is_successful:
            self.console.print(
                "\n[bold green]Migration completed successfully![/bold green]"
            )
        else:
            self.console.print(
                f"\n[bold yellow]Migration completed with issues:[/bold yellow]"
            )
            self.console.print(f"  • {report.failed} failed migrations")
            self.console.print(f"  • {report.verification_failed} verification failures")

    def save_report(self, report: MigrationReport, output_path: str) -> None:
        """
        Save report to JSON file

        Args:
            report: Migration report
            output_path: Output file path
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        report_data = report.model_dump()
        output.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

        logger.info(f"Report saved to: {output_path}")
        self.console.print(f"[green]Report saved to: {output_path}[/green]")

    def _get_status_color(self, status: MigrationStatus) -> str:
        """Get color for status"""
        color_map = {
            MigrationStatus.PENDING: "yellow",
            MigrationStatus.IN_PROGRESS: "blue",
            MigrationStatus.COMPLETED: "green",
            MigrationStatus.FAILED: "red",
            MigrationStatus.VERIFIED: "bright_green",
            MigrationStatus.VERIFICATION_FAILED: "orange1",
        }
        return color_map.get(status, "white")
