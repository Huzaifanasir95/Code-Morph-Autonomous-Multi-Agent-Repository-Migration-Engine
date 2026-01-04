"""
Docker Manager for Sandbox Environments

Manages Docker containers for isolated test execution.
Handles container lifecycle, resource limits, and cleanup.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import docker
from docker.errors import DockerException, NotFound
from docker.models.containers import Container

from src.test_sandbox.schemas.test_models import DockerConfig, SandboxEnvironment
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DockerManager:
    """Manages Docker containers for test sandboxing"""

    def __init__(self):
        """Initialize Docker client"""
        try:
            self.client = docker.from_env()
            logger.info("Docker client initialized")
        except DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise RuntimeError(
                "Docker is not available. Please ensure Docker Desktop is running."
            )

    def create_sandbox(
        self,
        config: Optional[DockerConfig] = None,
        volumes: Optional[Dict[str, str]] = None,
        environment: Optional[Dict[str, str]] = None,
    ) -> SandboxEnvironment:
        """
        Create a new sandbox environment

        Args:
            config: Docker configuration
            volumes: Volume mappings {host_path: container_path}
            environment: Environment variables

        Returns:
            SandboxEnvironment with container details
        """
        if config is None:
            config = DockerConfig()

        if volumes is None:
            volumes = {}

        if environment is None:
            environment = {}

        logger.info(f"Creating sandbox with image: {config.image}")

        try:
            # Pull image if not available
            try:
                self.client.images.get(config.image)
            except NotFound:
                logger.info(f"Pulling image: {config.image}")
                self.client.images.pull(config.image)

            # Create container
            container = self.client.containers.create(
                image=config.image,
                detach=True,
                mem_limit=config.memory_limit,
                nano_cpus=int(config.cpu_limit * 1e9),  # Convert to nanocores
                network_disabled=config.network_disabled,
                working_dir=config.working_dir,
                volumes=volumes,
                environment=environment,
                command="sleep infinity",  # Keep container running
            )

            sandbox = SandboxEnvironment(
                container_id=container.id,
                status="created",
                config=config,
                volumes=volumes,
                environment_vars=environment,
            )

            logger.info(f"Sandbox created: {container.id[:12]}")
            return sandbox

        except DockerException as e:
            logger.error(f"Failed to create sandbox: {e}")
            raise

    def start_sandbox(self, sandbox: SandboxEnvironment) -> None:
        """
        Start a sandbox container

        Args:
            sandbox: Sandbox environment to start
        """
        if not sandbox.container_id:
            raise ValueError("Sandbox has no container ID")

        try:
            container = self.client.containers.get(sandbox.container_id)
            container.start()
            sandbox.status = "running"
            logger.info(f"Sandbox started: {sandbox.container_id[:12]}")
        except DockerException as e:
            logger.error(f"Failed to start sandbox: {e}")
            raise

    def execute_command(
        self,
        sandbox: SandboxEnvironment,
        command: str,
        timeout: Optional[int] = None,
    ) -> tuple[int, str, str]:
        """
        Execute command in sandbox

        Args:
            sandbox: Sandbox environment
            command: Command to execute
            timeout: Execution timeout in seconds

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        if not sandbox.container_id:
            raise ValueError("Sandbox has no container ID")

        if timeout is None:
            timeout = sandbox.config.timeout_seconds

        logger.info(f"Executing in sandbox: {command[:50]}...")

        try:
            container = self.client.containers.get(sandbox.container_id)

            # Execute command
            exec_result = container.exec_run(
                cmd=command,
                stdout=True,
                stderr=True,
                demux=True,
            )

            exit_code = exec_result.exit_code
            stdout_bytes, stderr_bytes = exec_result.output

            stdout = stdout_bytes.decode("utf-8") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8") if stderr_bytes else ""

            logger.info(f"Command completed with exit code: {exit_code}")
            return exit_code, stdout, stderr

        except DockerException as e:
            logger.error(f"Failed to execute command: {e}")
            return 1, "", str(e)

    def copy_to_sandbox(
        self, sandbox: SandboxEnvironment, source_path: str, dest_path: str
    ) -> None:
        """
        Copy file to sandbox

        Args:
            sandbox: Sandbox environment
            source_path: Host file path
            dest_path: Container file path
        """
        if not sandbox.container_id:
            raise ValueError("Sandbox has no container ID")

        try:
            container = self.client.containers.get(sandbox.container_id)
            
            # Read file content
            source = Path(source_path)
            if not source.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")

            # Create tar archive in memory
            import tarfile
            import io

            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tar.add(source_path, arcname=Path(dest_path).name)

            tar_stream.seek(0)

            # Copy to container
            container.put_archive(Path(dest_path).parent, tar_stream)
            logger.info(f"Copied {source_path} to sandbox:{dest_path}")

        except DockerException as e:
            logger.error(f"Failed to copy file to sandbox: {e}")
            raise

    def stop_sandbox(self, sandbox: SandboxEnvironment, timeout: int = 10) -> None:
        """
        Stop a sandbox container

        Args:
            sandbox: Sandbox environment to stop
            timeout: Timeout for graceful shutdown
        """
        if not sandbox.container_id:
            return

        try:
            container = self.client.containers.get(sandbox.container_id)
            container.stop(timeout=timeout)
            sandbox.status = "stopped"
            logger.info(f"Sandbox stopped: {sandbox.container_id[:12]}")
        except NotFound:
            logger.warning(f"Container not found: {sandbox.container_id[:12]}")
        except DockerException as e:
            logger.error(f"Failed to stop sandbox: {e}")

    def destroy_sandbox(self, sandbox: SandboxEnvironment, force: bool = True) -> None:
        """
        Remove a sandbox container

        Args:
            sandbox: Sandbox environment to destroy
            force: Force removal even if running
        """
        if not sandbox.container_id:
            return

        try:
            container = self.client.containers.get(sandbox.container_id)
            container.remove(force=force)
            sandbox.status = "destroyed"
            logger.info(f"Sandbox destroyed: {sandbox.container_id[:12]}")
        except NotFound:
            logger.warning(f"Container not found: {sandbox.container_id[:12]}")
        except DockerException as e:
            logger.error(f"Failed to destroy sandbox: {e}")

    def cleanup_all(self, label: str = "code-morph") -> int:
        """
        Clean up all containers with specific label

        Args:
            label: Container label to filter by

        Returns:
            Number of containers cleaned up
        """
        try:
            containers = self.client.containers.list(all=True, filters={"label": label})
            count = len(containers)

            for container in containers:
                try:
                    container.remove(force=True)
                except DockerException as e:
                    logger.error(f"Failed to remove container {container.id[:12]}: {e}")

            logger.info(f"Cleaned up {count} containers")
            return count

        except DockerException as e:
            logger.error(f"Failed to cleanup containers: {e}")
            return 0

    def is_docker_available(self) -> bool:
        """
        Check if Docker is available and running

        Returns:
            True if Docker is available
        """
        try:
            self.client.ping()
            return True
        except DockerException:
            return False
