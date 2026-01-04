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

from src.agent_orchestration.orchestrator import RepositoryOrchestrator
from src.ast_engine.analyzers.api_detector import APIDetector
from src.ast_engine.analyzers.migration_plan_generator import MigrationPlanGenerator
from src.ast_engine.parsers.python_parser import PythonParser
from src.migration_engine.transformers.python_transformer import PythonTransformer
from src.test_sandbox.comparator import OutputComparator
from src.test_sandbox.docker_manager import DockerManager
from src.test_sandbox.schemas.test_models import TestResult, TestStatus, TestSuiteResult
from src.test_sandbox.test_executor import TestExecutor
from src.test_sandbox.test_generator import TestGenerator
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
def migrate(
    source: str = typer.Argument(..., help="Path to source file to migrate"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output path for migrated code"
    ),
    target: str = typer.Option("pytorch", "--target", "-t", help="Target framework"),
    framework: str = typer.Option(
        "tensorflow==1.15.0", "--framework", "-f", help="Source framework version"
    ),
    no_llm: bool = typer.Option(
        False, "--no-llm", help="Disable LLM-based transformations"
    ),
    plan: Optional[str] = typer.Option(
        None, "--plan", "-p", help="Use existing migration plan JSON file"
    ),
) -> None:
    """
    Migrate legacy code to modern framework

    Examples:
        code-morph migrate legacy_model.py --target pytorch
        code-morph migrate legacy_model.py --output migrated.py --target pytorch
        code-morph migrate legacy_model.py --plan migration_plan.json
    """
    console.print("\n[bold cyan]🔄 Code-Morph Migration Engine[/bold cyan]\n")

    source_path = Path(source)
    if not source_path.exists():
        console.print(f"[red]❌ Error: Source file not found: {source}[/red]")
        raise typer.Exit(1)

    # Ensure output directories exist
    ensure_output_dirs()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Step 1: Load or generate migration plan
        if plan:
            task1 = progress.add_task(f"[cyan]Loading migration plan from {plan}...", total=None)
            try:
                generator = MigrationPlanGenerator()
                migration_plan = generator.load_plan(plan)
                progress.update(task1, completed=True)
                console.print(f"✅ [green]Migration plan loaded[/green]")
            except Exception as e:
                progress.stop()
                console.print(f"[red]❌ Error loading plan: {e}[/red]")
                raise typer.Exit(1)
        else:
            task1 = progress.add_task("[cyan]Generating migration plan...", total=None)
            try:
                rules_file = get_rules_file("python")
                generator = MigrationPlanGenerator(rules_file)
                migration_plan = generator.generate_plan(str(source_path), target, framework)
                progress.update(task1, completed=True)
                console.print("✅ [green]Migration plan generated[/green]")
            except Exception as e:
                progress.stop()
                console.print(f"[red]❌ Error generating plan: {e}[/red]")
                raise typer.Exit(1)

        # Display plan summary
        console.print(f"\n📋 Migration Plan: {len(migration_plan.transformations)} transformations")
        console.print(f"   Complexity: [yellow]{migration_plan.estimated_complexity.value.upper()}[/yellow]")
        console.print(f"   Estimated Time: [cyan]{migration_plan.estimated_time_minutes} minutes[/cyan]")

        if migration_plan.warnings:
            console.print("\n[yellow]⚠️  Warnings:[/yellow]")
            for warning in migration_plan.warnings:
                console.print(f"   • {warning}")

        # Step 2: Execute migration
        task2 = progress.add_task("[cyan]Transforming code...", total=None)
        try:
            transformer = PythonTransformer(
                source_framework=framework.split("==")[0],
                target_framework=target,
                use_llm=not no_llm,
            )

            # Determine output path
            if output is None:
                output = str(settings.outputs_dir / "migrated_code" / source_path.name)

            # Perform transformation
            output_path = transformer.transform_file(
                str(source_path), migration_plan, output
            )

            progress.update(task2, completed=True)
            console.print(f"✅ [green]Code transformation complete[/green]")

        except Exception as e:
            progress.stop()
            console.print(f"[red]❌ Error during migration: {e}[/red]")
            logger.error(f"Migration failed: {e}", exc_info=True)
            raise typer.Exit(1)

    # Display results
    console.print(f"\n[bold green]✨ Migration Complete![/bold green]")
    console.print(f"\n📁 Migrated code saved to: [cyan]{output_path}[/cyan]")

    # Show preview of migrated code
    try:
        migrated_code = Path(output_path).read_text(encoding="utf-8")
        lines = migrated_code.split("\n")
        preview_lines = min(20, len(lines))

        console.print(f"\n[bold]Preview (first {preview_lines} lines):[/bold]")
        syntax = Syntax(
            "\n".join(lines[:preview_lines]),
            "python",
            theme="monokai",
            line_numbers=True,
        )
        console.print(syntax)

        if len(lines) > preview_lines:
            console.print(f"\n... and {len(lines) - preview_lines} more lines")

    except Exception as e:
        logger.error(f"Error displaying preview: {e}")

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


@app.command()
def repo_migrate(
    repository: str = typer.Argument(..., help="Path to repository to migrate"),
    source_framework: str = typer.Option(..., "--source", "-s", help="Source framework (e.g., tensorflow)"),
    target_framework: str = typer.Option(..., "--target", "-t", help="Target framework (e.g., pytorch)"),
    output: str = typer.Option("migrated", "--output", "-o", help="Output directory for migrated code"),
    include: Optional[str] = typer.Option(None, "--include", help="Include patterns (comma-separated, e.g., *.py,src/*)"),
    exclude: Optional[str] = typer.Option(None, "--exclude", help="Exclude patterns (comma-separated, e.g., tests/*,*_test.py)"),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Enable automated verification"),
    max_parallel: int = typer.Option(3, "--max-parallel", "-p", help="Maximum parallel migrations"),
    report_json: Optional[str] = typer.Option(None, "--report", "-r", help="Save JSON report to file"),
) -> None:
    """
    Migrate an entire repository autonomously with multi-agent orchestration.
    
    This command orchestrates a full repository migration by:
    1. Scanning the repository for files needing migration
    2. Resolving dependencies and determining migration order
    3. Creating batches for parallel execution
    4. Migrating files with rate limiting and error handling
    5. Verifying migrated code (if enabled)
    """
    console.print("\n[bold cyan]Code-Morph Repository Migration[/bold cyan]")
    console.print(f"Source: {repository}")
    console.print(f"Migration: {source_framework} -> {target_framework}")
    console.print(f"Output: {output}")
    console.print()
    
    # Parse include/exclude patterns
    include_patterns = [p.strip() for p in include.split(",")] if include else None
    exclude_patterns = [p.strip() for p in exclude.split(",")] if exclude else None
    
    # Initialize orchestrator
    orchestrator = RepositoryOrchestrator(
        output_dir=output,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        use_docker=False,  # Use local testing (all libraries installed)
        max_parallel=max_parallel,
    )
    
    try:
        # Run autonomous migration
        with console.status("[bold green]Migrating repository...", spinner="dots"):
            report = orchestrator.migrate_repository(
                repo_path=repository,
                source_framework=source_framework,
                target_framework=target_framework,
                verify=verify,
            )
        
        # Display report
        console.print()
        orchestrator.report_generator.print_report(report)
        
        # Save JSON report if requested
        if report_json:
            orchestrator.report_generator.save_report(report, report_json)
            console.print(f"\n[green]Report saved to: {report_json}")
        
        # Exit with appropriate code
        if report.failed > 0:
            console.print("\n[yellow]Migration completed with errors")
            raise typer.Exit(code=1)
        else:
            console.print("\n[green]Migration completed successfully!")
            raise typer.Exit(code=0)
            
    except Exception as e:
        console.print(f"\n[red]Error: {e}")
        logger.exception("Repository migration failed")
        raise typer.Exit(code=1)


@app.command()
def verify(
    legacy_file: str = typer.Argument(..., help="Path to legacy code file"),
    migrated_file: str = typer.Argument(..., help="Path to migrated code file"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file for verification report (JSON)"
    ),
    requirements: Optional[str] = typer.Option(
        None, "--requirements", "-r", help="Comma-separated list of dependencies"
    ),
    no_docker: bool = typer.Option(
        False, "--no-docker", help="Skip Docker sandboxing (run locally)"
    ),
) -> None:
    """
    🔬 Verify behavioral equivalence between legacy and migrated code.

    Generates tests, executes them in sandboxed environments,
    and compares outputs to prove zero logical drift.
    """
    console.print(
        Panel.fit(
            "[bold cyan]Code-Morph Verification Engine[/bold cyan]\n"
            "Proving zero logical drift...",
            border_style="cyan",
        )
    )

    legacy_path = Path(legacy_file)
    migrated_path = Path(migrated_file)

    if not legacy_path.exists():
        console.print(f"[red]ERROR: Legacy file not found: {legacy_file}[/red]")
        raise typer.Exit(1)

    if not migrated_path.exists():
        console.print(f"[red]ERROR: Migrated file not found: {migrated_file}[/red]")
        raise typer.Exit(1)

    # Parse requirements
    req_list = []
    if requirements:
        req_list = [r.strip() for r in requirements.split(",")]

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Step 1: Generate tests
            progress.add_task("Generating tests with LLM...", total=None)
            test_generator = TestGenerator()

            # Read legacy code
            legacy_code = legacy_path.read_text(encoding="utf-8")
            
            # Generate tests
            test_code = test_generator.generate_tests(
                code=legacy_code,
                file_path=str(legacy_path),
                framework="python",
            )
            
            # Save tests
            test_output_dir = Path("outputs/verification/tests")
            test_output_dir.mkdir(parents=True, exist_ok=True)
            legacy_tests = test_output_dir / f"test_{legacy_path.stem}.py"
            test_generator.save_tests(test_code, str(legacy_tests))
            
            console.print(f"[green]Generated tests: {legacy_tests}[/green]")

            # Step 2: Check Docker availability
            if not no_docker:
                progress.add_task("Checking Docker availability...", total=None)
                docker_mgr = DockerManager()
                if not docker_mgr.is_docker_available():
                    console.print(
                        "[yellow]WARNING: Docker not available, falling back to local execution[/yellow]"
                    )
                    no_docker = True

            # Step 3: Execute tests
            if no_docker:
                console.print(
                    "[yellow]WARNING: Running tests locally (no sandboxing)[/yellow]"
                )
                # Local execution (simplified)
                legacy_results = _run_tests_locally(legacy_path, legacy_tests, req_list)
                migrated_results = _run_tests_locally(
                    migrated_path, legacy_tests, req_list
                )
            else:
                progress.add_task("Executing tests in sandbox...", total=None)
                test_executor = TestExecutor()

                legacy_results = test_executor.execute_tests(
                    test_file=str(legacy_tests),
                    code_file=str(legacy_path),
                    requirements=req_list,
                )

                migrated_results = test_executor.execute_tests(
                    test_file=str(legacy_tests),
                    code_file=str(migrated_path),
                    requirements=req_list,
                )

            # Step 4: Compare results
            progress.add_task("Comparing outputs...", total=None)
            comparator = OutputComparator()
            comparison = comparator.compare_test_results(
                legacy_results, migrated_results
            )

        # Display results
        console.print("\n[bold]Verification Results:[/bold]\n")

        # Create results table
        results_table = Table(show_header=True, header_style="bold magenta")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Legacy", style="yellow")
        results_table.add_column("Migrated", style="green")

        results_table.add_row(
            "Tests Passed",
            f"{legacy_results.passed}/{legacy_results.total_tests}",
            f"{migrated_results.passed}/{migrated_results.total_tests}",
        )
        results_table.add_row(
            "Execution Time",
            f"{legacy_results.duration_ms/1000:.2f}s",
            f"{migrated_results.duration_ms/1000:.2f}s",
        )

        console.print(results_table)

        # Comparison results
        console.print(f"\n[bold]Behavioral Equivalence:[/bold]")
        console.print(f"  Similarity Score: {comparison.similarity_score:.1%}")

        if comparison.are_equivalent:
            console.print(
                "\n[bold green]VERIFIED: Zero logical drift confirmed![/bold green]"
            )
        else:
            console.print(
                f"\n[bold red]DIFFERENCES DETECTED ({len(comparison.differences or [])} issues)[/bold red]"
            )
            if comparison.differences:
                for diff in comparison.differences[:10]:
                    console.print(f"  • {diff}")

        # Save report if requested
        if output:
            report_data = {
                "legacy_results": {
                    "file": str(legacy_path),
                    "passed": legacy_results.passed,
                    "total": legacy_results.total_tests,
                    "execution_time": legacy_results.duration_ms / 1000,
                },
                "migrated_results": {
                    "file": str(migrated_path),
                    "passed": migrated_results.passed,
                    "total": migrated_results.total_tests,
                    "execution_time": migrated_results.duration_ms / 1000,
                },
                "comparison": {
                    "are_equivalent": comparison.are_equivalent,
                    "similarity_score": comparison.similarity_score,
                    "differences": comparison.differences,
                },
            }

            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report_data, indent=2))
            console.print(f"\n[green]Report saved to: {output}[/green]")

    except Exception as e:
        console.print(f"\n[red]ERROR: Verification failed: {e}[/red]")
        logger.exception("Verification error")
        raise typer.Exit(1)


def _run_tests_locally(code_file: Path, test_file: str, requirements: list) -> "TestSuiteResult":
    """
    Run tests locally without Docker (simplified)

    Args:
        code_file: Code file path
        test_file: Test file path
        requirements: Package requirements

    Returns:
        TestSuiteResult
    """
    import subprocess
    import shutil
    import os
    from src.test_sandbox.schemas.test_models import TestResult, TestSuiteResult

    try:
        # Copy code file to test directory so imports work
        test_dir = Path(test_file).parent
        code_filename = code_file.name
        target_code_path = test_dir / code_filename
        shutil.copy2(code_file, target_code_path)
        
        # Set PYTHONPATH to include test directory
        env = os.environ.copy()
        env["PYTHONPATH"] = str(test_dir) + os.pathsep + env.get("PYTHONPATH", "")
        
        # Run pytest (override config to avoid pytest-cov dependency)
        result = subprocess.run(
            ["pytest", test_file, "-v", "--tb=short", "--override-ini=addopts="],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            cwd=str(test_dir),
        )

        # Parse output (simplified)
        passed = result.returncode == 0
        total = 1  # Simplified
        
        # Log errors for debugging
        if not passed:
            logger.error(f"Test execution failed for {code_file.name}")
            logger.error(f"STDOUT: {result.stdout[:500]}")
            logger.error(f"STDERR: {result.stderr[:500]}")

        return TestSuiteResult(
            file_path=str(test_file),
            framework="python",
            total_tests=total,
            passed=total if passed else 0,
            failed=0 if passed else total,
            duration_ms=0.0,
            tests=[
                TestResult(
                    test_name="local_test",
                    status=TestStatus.PASSED if passed else TestStatus.FAILED,
                    duration_ms=0.0,
                    error=result.stderr if not passed else None,
                    output=result.stdout,
                )
            ],
        )

    except Exception as e:
        return TestSuiteResult(
            file_path=str(test_file),
            framework="python",
            total_tests=1,
            passed=0,
            failed=1,
            duration_ms=0.0,
            tests=[
                TestResult(
                    test_name="local_test",
                    status=TestStatus.ERROR,
                    duration_ms=0.0,
                    error=str(e),
                )
            ],
        )


if __name__ == "__main__":
    app()
