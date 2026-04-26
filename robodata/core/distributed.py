"""Distributed computing framework for scalable data processing."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from robodata.core.async_utils import AsyncConfig
from robodata.core.exceptions import RobodataError

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class DistributedError(RobodataError):
    """Distributed computing specific error."""
    pass


@dataclass
class NodeInfo:
    """Information about a compute node."""
    
    node_id: str
    host: str
    port: int
    status: str = "active"
    capabilities: Dict[str, Any] = field(default_factory=dict)
    load: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "status": self.status,
            "capabilities": self.capabilities,
            "load": self.load,
            "last_heartbeat": self.last_heartbeat.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeInfo":
        """Create from dictionary."""
        node = cls(
            node_id=data["node_id"],
            host=data["host"],
            port=data["port"],
            status=data.get("status", "active"),
            capabilities=data.get("capabilities", {}),
            load=data.get("load", 0.0),
        )
        if "last_heartbeat" in data:
            node.last_heartbeat = datetime.fromisoformat(data["last_heartbeat"])
        return node


@dataclass
class TaskInfo:
    """Information about a distributed task."""
    
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    status: str = "pending"
    assigned_node: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "status": self.status,
            "assigned_node": self.assigned_node,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskInfo":
        """Create from dictionary."""
        task = cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            payload=data["payload"],
            status=data.get("status", "pending"),
            assigned_node=data.get("assigned_node"),
            result=data.get("result"),
            error=data.get("error"),
        )
        if "created_at" in data:
            task.created_at = datetime.fromisoformat(data["created_at"])
        if "started_at" in data and data["started_at"]:
            task.started_at = datetime.fromisoformat(data["started_at"])
        if "completed_at" in data and data["completed_at"]:
            task.completed_at = datetime.fromisoformat(data["completed_at"])
        return task


class DistributedNode(ABC):
    """Abstract base class for distributed compute nodes."""
    
    def __init__(self, node_id: str, host: str, port: int):
        """Initialize node.
        
        Args:
            node_id: Unique node identifier.
            host: Node host address.
            port: Node port.
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.status = "active"
        self.capabilities: Dict[str, Any] = {}
        self.load = 0.0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{node_id}")
    
    @abstractmethod
    async def start(self) -> None:
        """Start the node."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the node."""
        pass
    
    @abstractmethod
    async def execute_task(self, task: TaskInfo) -> Any:
        """Execute a task.
        
        Args:
            task: Task to execute.
            
        Returns:
            Task result.
        """
        pass
    
    async def get_status(self) -> NodeInfo:
        """Get node status.
        
        Returns:
            Node information.
        """
        return NodeInfo(
            node_id=self.node_id,
            host=self.host,
            port=self.port,
            status=self.status,
            capabilities=self.capabilities,
            load=self.load,
        )
    
    async def update_load(self, load: float) -> None:
        """Update node load.
        
        Args:
            load: Current load (0.0 to 1.0).
        """
        self.load = max(0.0, min(1.0, load))


class LocalNode(DistributedNode):
    """Local compute node for testing and development."""
    
    def __init__(self, node_id: str, host: str = "localhost", port: int = 0):
        """Initialize local node.
        
        Args:
            node_id: Node identifier.
            host: Host address.
            port: Port number.
        """
        super().__init__(node_id, host, port)
        self.task_handlers: Dict[str, Any] = {}
        self.running = False
    
    def register_handler(self, task_type: str, handler: Any) -> None:
        """Register task handler.
        
        Args:
            task_type: Task type.
            handler: Handler function or class.
        """
        self.task_handlers[task_type] = handler
        self.logger.debug(f"Registered handler for task type: {task_type}")
    
    async def start(self) -> None:
        """Start the node."""
        self.running = True
        self.status = "active"
        self.logger.info(f"Local node {self.node_id} started")
    
    async def stop(self) -> None:
        """Stop the node."""
        self.running = False
        self.status = "stopped"
        self.logger.info(f"Local node {self.node_id} stopped")
    
    async def execute_task(self, task: TaskInfo) -> Any:
        """Execute a task locally.
        
        Args:
            task: Task to execute.
            
        Returns:
            Task result.
        """
        if task.task_type not in self.task_handlers:
            raise DistributedError(f"No handler for task type: {task.task_type}")
        
        handler = self.task_handlers[task.task_type]
        self.logger.info(f"Executing task {task.task_id} of type {task.task_type}")
        
        try:
            # Update load
            await self.update_load(0.8)
            
            # Execute handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**task.payload)
            else:
                result = handler(**task.payload)
            
            self.logger.info(f"Task {task.task_id} completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Task {task.task_id} failed: {e}")
            raise
        finally:
            # Reset load
            await self.update_load(0.0)


class ClusterManager:
    """Manager for distributed compute cluster."""
    
    def __init__(self, config: AsyncConfig):
        """Initialize cluster manager.
        
        Args:
            config: Async configuration.
        """
        self.config = config
        self.nodes: Dict[str, DistributedNode] = {}
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_queue: List[str] = []
        self.running = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def start(self) -> None:
        """Start the cluster manager."""
        self.running = True
        self.logger.info("Cluster manager started")
    
    async def stop(self) -> None:
        """Stop the cluster manager."""
        self.running = False
        
        # Stop all nodes
        for node in self.nodes.values():
            await node.stop()
        
        self.logger.info("Cluster manager stopped")
    
    async def add_node(self, node: DistributedNode) -> None:
        """Add a node to the cluster.
        
        Args:
            node: Node to add.
        """
        await node.start()
        self.nodes[node.node_id] = node
        self.logger.info(f"Added node {node.node_id} to cluster")
    
    async def remove_node(self, node_id: str) -> None:
        """Remove a node from the cluster.
        
        Args:
            node_id: Node identifier.
        """
        if node_id in self.nodes:
            await self.nodes[node_id].stop()
            del self.nodes[node_id]
            self.logger.info(f"Removed node {node_id} from cluster")
    
    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        task_id: Optional[str] = None,
    ) -> str:
        """Submit a task to the cluster.
        
        Args:
            task_type: Type of task.
            payload: Task payload.
            task_id: Optional task identifier.
            
        Returns:
            Task identifier.
        """
        if not task_id:
            task_id = str(uuid.uuid4())
        
        task = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
        )
        
        self.tasks[task_id] = task
        self.task_queue.append(task_id)
        
        self.logger.info(f"Submitted task {task_id} of type {task_type}")
        
        # Trigger task processing
        asyncio.create_task(self._process_tasks())
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """Get task status.
        
        Args:
            task_id: Task identifier.
            
        Returns:
            Task information or None.
        """
        return self.tasks.get(task_id)
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status.
        
        Returns:
            Cluster status information.
        """
        node_statuses = {}
        for node_id, node in self.nodes.items():
            node_statuses[node_id] = await node.get_status()
        
        task_counts = {
            "pending": sum(1 for t in self.tasks.values() if t.status == "pending"),
            "running": sum(1 for t in self.tasks.values() if t.status == "running"),
            "completed": sum(1 for t in self.tasks.values() if t.status == "completed"),
            "failed": sum(1 for t in self.tasks.values() if t.status == "failed"),
        }
        
        return {
            "nodes": node_statuses,
            "tasks": task_counts,
            "queue_size": len(self.task_queue),
        }
    
    async def _process_tasks(self) -> None:
        """Process pending tasks."""
        while self.task_queue and self.running:
            # Find available node
            available_node = self._find_available_node()
            if not available_node:
                await asyncio.sleep(0.1)
                continue
            
            # Get next task
            task_id = self.task_queue.pop(0)
            task = self.tasks[task_id]
            
            # Assign task to node
            task.assigned_node = available_node.node_id
            task.status = "running"
            task.started_at = datetime.now()
            
            self.logger.info(f"Assigned task {task_id} to node {available_node.node_id}")
            
            # Execute task asynchronously
            asyncio.create_task(self._execute_task(available_node, task))
    
    def _find_available_node(self) -> Optional[DistributedNode]:
        """Find an available node.
        
        Returns:
            Available node or None.
        """
        for node in self.nodes.values():
            if node.status == "active" and node.load < 0.8:
                return node
        return None
    
    async def _execute_task(self, node: DistributedNode, task: TaskInfo) -> None:
        """Execute a task on a node.
        
        Args:
            node: Node to execute on.
            task: Task to execute.
        """
        try:
            result = await node.execute_task(task)
            
            task.result = result
            task.status = "completed"
            task.completed_at = datetime.now()
            
            self.logger.info(f"Task {task.task_id} completed on node {node.node_id}")
            
        except Exception as e:
            task.error = str(e)
            task.status = "failed"
            task.completed_at = datetime.now()
            
            self.logger.error(f"Task {task.task_id} failed on node {node.node_id}: {e}")


class DistributedExecutor:
    """High-level interface for distributed execution."""
    
    def __init__(self, cluster: ClusterManager):
        """Initialize executor.
        
        Args:
            cluster: Cluster manager.
        """
        self.cluster = cluster
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def map(
        self,
        func: str,
        inputs: List[Dict[str, Any]],
        task_type: str = "map",
    ) -> List[Any]:
        """Map function over inputs distributedly.
        
        Args:
            func: Function name to execute.
            inputs: List of input dictionaries.
            task_type: Task type identifier.
            
        Returns:
            List of results.
        """
        task_ids = []
        
        # Submit all tasks
        for i, input_data in enumerate(inputs):
            payload = {
                "func": func,
                "input": input_data,
                "index": i,
            }
            task_id = await self.cluster.submit_task(task_type, payload)
            task_ids.append(task_id)
        
        # Wait for completion
        results = [None] * len(task_ids)
        
        for task_id in task_ids:
            while True:
                task = await self.cluster.get_task_status(task_id)
                if task and task.status in ["completed", "failed"]:
                    if task.status == "completed":
                        results[task.payload["index"]] = task.result
                    else:
                        self.logger.error(f"Task {task_id} failed: {task.error}")
                    break
                await asyncio.sleep(0.1)
        
        return results
    
    async def reduce(
        self,
        func: str,
        inputs: List[Any],
        task_type: str = "reduce",
    ) -> Any:
        """Reduce inputs using distributed computation.
        
        Args:
            func: Function name to execute.
            inputs: List of inputs to reduce.
            task_type: Task type identifier.
            
        Returns:
            Reduced result.
        """
        # For simplicity, execute reduce on a single node
        payload = {
            "func": func,
            "inputs": inputs,
        }
        
        task_id = await self.cluster.submit_task(task_type, payload)
        
        while True:
            task = await self.cluster.get_task_status(task_id)
            if task and task.status in ["completed", "failed"]:
                if task.status == "completed":
                    return task.result
                else:
                    raise DistributedError(f"Reduce task failed: {task.error}")
            await asyncio.sleep(0.1)
    
    async def scatter_gather(
        self,
        tasks: List[Dict[str, Any]],
        task_type: str = "scatter_gather",
    ) -> List[Any]:
        """Scatter tasks and gather results.
        
        Args:
            tasks: List of task specifications.
            task_type: Task type identifier.
            
        Returns:
            List of results.
        """
        task_ids = []
        
        # Submit all tasks
        for task_spec in tasks:
            task_id = await self.cluster.submit_task(task_type, task_spec)
            task_ids.append(task_id)
        
        # Gather results
        results = []
        
        for task_id in task_ids:
            while True:
                task = await self.cluster.get_task_status(task_id)
                if task and task.status in ["completed", "failed"]:
                    if task.status == "completed":
                        results.append(task.result)
                    else:
                        self.logger.error(f"Task {task_id} failed: {task.error}")
                    break
                await asyncio.sleep(0.1)
        
        return results


# Utility functions
async def create_local_cluster(
    num_nodes: int = 4,
    config: Optional[AsyncConfig] = None,
) -> ClusterManager:
    """Create a local cluster for testing.
    
    Args:
        num_nodes: Number of local nodes.
        config: Async configuration.
        
    Returns:
        Cluster manager.
    """
    if config is None:
        config = AsyncConfig()
    
    cluster = ClusterManager(config)
    await cluster.start()
    
    # Create local nodes
    for i in range(num_nodes):
        node = LocalNode(f"local-node-{i}")
        await cluster.add_node(node)
    
    return cluster


async def benchmark_cluster(
    cluster: ClusterManager,
    num_tasks: int = 100,
    task_size: int = 1000,
) -> Dict[str, Any]:
    """Benchmark cluster performance.
    
    Args:
        cluster: Cluster to benchmark.
        num_tasks: Number of tasks to run.
        task_size: Size of each task.
        
    Returns:
        Benchmark results.
    """
    import time
    
    # Create dummy tasks
    tasks = [
        {
            "func": "dummy_compute",
            "size": task_size,
            "duration": 0.1,  # 100ms per task
        }
        for _ in range(num_tasks)
    ]
    
    # Register dummy handler on all nodes
    async def dummy_compute(size: int, duration: float) -> int:
        await asyncio.sleep(duration)
        return size * 2
    
    for node in cluster.nodes.values():
        if isinstance(node, LocalNode):
            node.register_handler("dummy_compute", dummy_compute)
    
    # Run benchmark
    start_time = time.time()
    
    executor = DistributedExecutor(cluster)
    results = await executor.scatter_gather(tasks, "benchmark")
    
    end_time = time.time()
    
    return {
        "num_tasks": num_tasks,
        "task_size": task_size,
        "total_time": end_time - start_time,
        "tasks_per_second": num_tasks / (end_time - start_time),
        "successful_tasks": len(results),
        "cluster_status": await cluster.get_cluster_status(),
    }
