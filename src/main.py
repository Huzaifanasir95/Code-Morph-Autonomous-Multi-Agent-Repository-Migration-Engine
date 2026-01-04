"""
Code-Morph CLI Application

Main entry point for the Code-Morph migration engine.
Provides commands for analyzing and migrating legacy codebases.
"""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from src.ast_engine.analyzers.api_detector import APIDetector
from src.ast_engine.analyzers.migration_plan_generator import MigrationPlanGenerator
from src.ast_engine.parsers.python_parser import PythonParser
from src.utils.config import ensure_output_dirs, get_rules_file, settings
from src.utils.file_handler import FileHandler
from src.utils.logger import get_logger

# Initialize CLI app
app = typer.Typer(
    name="code-morph",
    help="🔄 Autonomous Multi-Agent Repository Migration Engine",
    add_completion=False,
)

# Rich console for beautiful output
console = Console()
logger = get_logger(__name__)


@app.command()
def analyze(
    source: str = typer.Argument(..., help="Path to source file or directory"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file for migration plan (JSON)"
    ),
    target: str = typer.Option("pytorch", "--target", "-t", help="Target framework"),
    framework: str = typer.Option(
        "tensorflow==1.15.0", "--framework", "-f", help="Source framework version"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """
    Analyze legacy code and generate migration plan

    Examples:
        code-morph analyze legacy_model.py
        code-morph analyze legacy_model.py --output plan.json --target pytorch
    """
    console.print("\n[bold cyan]🔍 Code-Morph Analysis Engine[/bold cyan]\n")

    source_path = Path(source)
    if not source_path.exists():
        console.print(f"[red]❌ Error: Source file not found: {source}[/red]")
        raise typer.Exit(1)

    # Ensure output directories exist
    ensure_output_dirs()

    # Get rules file for API detection
    rules_file = get_rules_file("python")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Step 1: Parse file
        task1 = progress.add_task("[cyan]Parsing source code...", total=None)
        try:
            parser = PythonParser(str(source_path))
            analysis = parser.parse()
            progress.update(task1, completed=True)
            console.print("✅ [green]Source code parsed successfully[/green]")
        except Exception as e:
            progress.stop()
            console.print(f"[red]❌ Error parsing file: {e}[/red]")
            raise typer.Exit(1)

        # Step 2: Analyze dependencies
        task2 = progress.add_task("[cyan]Analyzing dependencies...", total=None)
        console.print(f"📦 Found {len(analysis.dependencies)} dependencies")
        progress.update(task2, completed=True)

        # Step 3: Detect deprecated APIs
        task3 = progress.add_task("[cyan]Detecting deprecated APIs...", total=None)
        if rules_file:
            detector = APIDetector(rules_file)
            framework_key = "tensorflow_v1" if "tensorflow" in framework else framework
            deprecated = detector.detect(analysis, framework=framework_key)
            analysis.deprecated_apis = deprecated
            progress.update(task3, completed=True)
            console.print(f"⚠️  Found {len(deprecated)} deprecated API usages")
        else:
            progress.update(task3, completed=True)
            console.print("⚠️  No rules file found - skipping API detection")

        # Step 4: Generate migration plan
        task4 = progress.add_task("[cyan]Generating migration plan...", total=None)
        generator = MigrationPlanGenerator(rules_file)
        plan = generator.generate_plan(str(source_path), target, framework)
        progress.update(task4, completed=True)

    # Display results
    _display_analysis(analysis, plan)

    # Save migration plan if output specified
    if output:
        generator.save_plan(plan, output)
        console.print(f"\n✅ [green]Migration plan saved to: {output}[/green]")
    else:
        # Save to default location
        output_path = settings.outputs_dir / "migration_plans" / f"{source_path.stem}_plan.json"
        generator.save_plan(plan, str(output_path))
        console.print(f"\n✅ [green]Migration plan saved to: {output_path}[/green]")

    # Show migration plan summary
    _display_migration_plan(plan)

    console.print("\n[bold green]✨ Analysis complete![/bold green]\n")


@app.command()
def info(
    file: str = typer.Argument(..., help="Path to source file"),
) -> None:
    """
    Display detailed information about a source file

    Examples:
        code-morph info legacy_model.py
    """
    console.print("\n[bold cyan]📊 File Information[/bold cyan]\n")

    file_path = Path(file)
    if not file_path.exists():
        console.print(f"[red]❌ Error: File not found: {file}[/red]")
        raise typer.Exit(1)

    try:
        parser = PythonParser(str(file_path))
        analysis = parser.parse()
    except Exception as e:
        console.print(f"[red]❌ Error parsing file: {e}[/red]")
        raise typer.Exit(1)

    # Display file info
    table = Table(title=f"Analysis: {file_path.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Lines", str(analysis.total_lines))
    table.add_row("Functions", str(len(analysis.functions)))
    table.add_row("Classes", str(len(analysis.classes)))
    table.add_row("Imports", str(len(analysis.imports)))
    table.add_row("Dependencies", str(len(analysis.dependencies)))
    table.add_row("Complexity Score", f"{analysis.complexity_score:.1f}/10")

    console.print(table)

    # Display imports
    if analysis.imports:
        console.print("\n[bold]📦 Imports:[/bold]")
        for imp in analysis.imports[:10]:  # Show first 10
            console.print(f"  • {imp.module}")

    # Display functions
    if analysis.functions:
        console.print("\n[bold]⚡ Functions:[/bold]")
        for func in analysis.functions[:10]:  # Show first 10
            params = ", ".join(func.parameters)
            console.print(f"  • {func.name}({params})")

    console.print()


@app.command()
def version() -> None:
    """Display Code-Morph version"""
    from src import __version__

    console.print(f"\n[bold cyan]Code-Morph[/bold cyan] version [green]{__version__}[/green]\n")


def _display_analysis(analysis, plan) -> None:
    """Display analysis results in rich format"""
    # Create summary panel
    summary = f"""
[bold]File:[/bold] {analysis.file_path}
[bold]Lines of Code:[/bold] {analysis.total_lines}
[bold]Functions:[/bold] {len(analysis.functions)}
[bold]Classes:[/bold] {len(analysis.classes)}
[bold]Complexity:[/bold] {analysis.complexity_score:.1f}/10
    """

    console.print(Panel(summary.strip(), title="📊 Analysis Summary", border_style="cyan"))

    # Display deprecated APIs if found
    if analysis.deprecated_apis:
        console.print("\n[bold red]⚠️  Deprecated APIs Found:[/bold red]")
        for api in analysis.deprecated_apis[:5]:  # Show first 5
            console.print(f"  Line {api.line_number}: [yellow]{api.api_name}[/yellow]")
            console.print(f"    → {api.suggestion}")


def _display_migration_plan(plan) -> None:
    """Display migration plan in rich format"""
    console.print("\n[bold cyan]📋 Migration Plan[/bold cyan]\n")

    # Create plan table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Attribute", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Source Framework", plan.source_framework)
    table.add_row("Target Framework", plan.target_framework)
    table.add_row("Complexity", plan.estimated_complexity.value.upper())
    table.add_row("Confidence Score", f"{plan.confidence_score:.0%}")
    table.add_row("Estimated Time", f"{plan.estimated_time_minutes} minutes")
    table.add_row("Transformations", str(len(plan.transformations)))
    table.add_row("Manual Review", "Yes" if plan.manual_review_required else "No")

    console.print(table)

    # Display warnings if any
    if plan.warnings:
        console.print("\n[bold yellow]⚠️  Warnings:[/bold yellow]")
        for warning in plan.warnings:
            console.print(f"  • {warning}")

    # Display transformations summary
    if plan.transformations:
        console.print(f"\n[bold]🔧 Transformations ({len(plan.transformations)}):[/bold]")
        for i, transform in enumerate(plan.transformations[:5], 1):  # Show first 5
            console.print(f"  {i}. {transform.description}")
            if i == 5 and len(plan.transformations) > 5:
                console.print(f"  ... and {len(plan.transformations) - 5} more")
                break


if __name__ == "__main__":
    app()
