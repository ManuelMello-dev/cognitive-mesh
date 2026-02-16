"""
Always-On Orchestrator
Manages system lifecycle, fault tolerance, checkpointing, and auto-recovery
"""
import logging
import time
import threading
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """Information about a managed task"""
    task_id: str
    name: str
    fn: Optional[Callable] = None
    thread: Optional[threading.Thread] = None
    running: bool = False
    restart_count: int = 0
    last_start: Optional[datetime] = None
    last_error: Optional[str] = None

    def to_dict(self):
        return {
            'task_id': self.task_id,
            'name': self.name,
            'running': self.running,
            'restart_count': self.restart_count,
            'last_start': self.last_start.isoformat() if self.last_start else None,
            'last_error': self.last_error
        }


class AlwaysOnOrchestrator:
    """
    Manages always-on operation with:
    - Task management and monitoring
    - Automatic restart on failure
    - Periodic checkpointing
    - Health monitoring
    - Graceful shutdown
    """

    def __init__(
        self,
        checkpoint_interval: int = 300,
        auto_restart: bool = True,
        max_restarts: int = 10,
        health_check_interval: int = 60
    ):
        self.checkpoint_interval = checkpoint_interval
        self.auto_restart = auto_restart
        self.max_restarts = max_restarts
        self.health_check_interval = health_check_interval

        # Task management
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_counter = 0

        # System state
        self.running = False
        self.start_time: Optional[datetime] = None
        self.checkpoint_callbacks: List[Callable] = []
        self.last_checkpoint: Optional[datetime] = None

        # Health tracking
        self.health_history: List[Dict[str, Any]] = []
        self.error_log: List[Dict[str, Any]] = []

        logger.info(
            f"AlwaysOnOrchestrator initialized: "
            f"checkpoint_interval={checkpoint_interval}s, "
            f"auto_restart={auto_restart}"
        )

    def register_task(
        self,
        name: str,
        fn: Callable,
        auto_start: bool = False
    ) -> str:
        """Register a managed task"""
        task_id = f"task_{self.task_counter}"
        self.task_counter += 1

        task = TaskInfo(
            task_id=task_id,
            name=name,
            fn=fn
        )

        self.tasks[task_id] = task
        logger.info(f"Registered task: {name} (id: {task_id})")

        if auto_start:
            self.start_task(task_id)

        return task_id

    def start_task(self, task_id: str):
        """Start a managed task in a thread"""
        if task_id not in self.tasks:
            logger.error(f"Unknown task: {task_id}")
            return

        task = self.tasks[task_id]

        if task.running:
            logger.warning(f"Task {task.name} is already running")
            return

        def _run_with_recovery():
            while self.running and task.restart_count < self.max_restarts:
                try:
                    task.running = True
                    task.last_start = datetime.now()
                    logger.info(f"Starting task: {task.name}")

                    if task.fn:
                        task.fn()

                    task.running = False
                    break  # Normal exit

                except Exception as e:
                    task.running = False
                    task.last_error = str(e)
                    task.restart_count += 1

                    self.error_log.append({
                        'task_id': task_id,
                        'task_name': task.name,
                        'error': str(e),
                        'restart_count': task.restart_count,
                        'timestamp': datetime.now().isoformat()
                    })

                    logger.error(
                        f"Task {task.name} failed (attempt {task.restart_count}): {e}"
                    )

                    if self.auto_restart and task.restart_count < self.max_restarts:
                        wait_time = min(2 ** task.restart_count, 60)
                        logger.info(f"Restarting {task.name} in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Task {task.name} exceeded max restarts")
                        break

        thread = threading.Thread(target=_run_with_recovery, daemon=True)
        task.thread = thread
        thread.start()

    def stop_task(self, task_id: str):
        """Stop a managed task"""
        if task_id in self.tasks:
            self.tasks[task_id].running = False
            logger.info(f"Stopped task: {self.tasks[task_id].name}")

    def stop_all(self):
        """Stop all managed tasks"""
        self.running = False
        for task_id in self.tasks:
            self.stop_task(task_id)
        logger.info("All tasks stopped")

    def register_checkpoint_callback(self, callback: Callable):
        """Register a function to be called during checkpoints"""
        self.checkpoint_callbacks.append(callback)

    def checkpoint(self):
        """Perform a checkpoint"""
        self.last_checkpoint = datetime.now()

        for callback in self.checkpoint_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Checkpoint callback failed: {e}")

        logger.info("Checkpoint completed")

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        uptime = 0
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            'running': self.running,
            'uptime_seconds': uptime,
            'tasks': {
                tid: task.to_dict()
                for tid, task in self.tasks.items()
            },
            'total_errors': len(self.error_log),
            'last_checkpoint': self.last_checkpoint.isoformat() if self.last_checkpoint else None,
            'recent_errors': self.error_log[-5:]
        }

    def save_state(self):
        """Save orchestrator state"""
        try:
            state = {
                'tasks': {tid: t.to_dict() for tid, t in self.tasks.items()},
                'error_log': self.error_log[-100:],
                'health_history': self.health_history[-100:]
            }
            filepath = '/tmp/orchestrator_state.json'
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            logger.debug("Orchestrator state saved")
        except Exception as e:
            logger.error(f"Failed to save orchestrator state: {e}")

    def load_state(self):
        """Load orchestrator state"""
        try:
            filepath = '/tmp/orchestrator_state.json'
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    state = json.load(f)
                self.error_log = state.get('error_log', [])
                self.health_history = state.get('health_history', [])
                logger.info("Orchestrator state loaded")
        except Exception as e:
            logger.warning(f"Could not load orchestrator state: {e}")
