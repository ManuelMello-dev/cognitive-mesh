"""
Universal Self-Evolving Intelligent System
Combines continuous learning, self-writing code, always-on operation
Domain-agnostic framework for autonomous intelligent systems
"""
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import json
import sys
import os

# Import our custom components
from continuous_learning_engine import ContinuousLearningEngine
from self_writing_engine import SelfEvolvingSystem
from always_on_orchestrator import AlwaysOnOrchestrator

logger = logging.getLogger(__name__)


class DataStreamAdapter:
    """Generic adapter for any data stream"""
    
    def __init__(self, stream_id: str, fetch_fn: Callable):
        self.stream_id = stream_id
        self.fetch_fn = fetch_fn
        self.last_fetch = None
        self.fetch_count = 0
        self.error_count = 0
    
    def fetch(self) -> Optional[Dict[str, Any]]:
        """Fetch data from stream"""
        try:
            data = self.fetch_fn()
            self.last_fetch = datetime.now()
            self.fetch_count += 1
            return data
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error fetching from {self.stream_id}: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics"""
        return {
            'stream_id': self.stream_id,
            'fetch_count': self.fetch_count,
            'error_count': self.error_count,
            'last_fetch': self.last_fetch.isoformat() if self.last_fetch else None,
            'error_rate': self.error_count / max(self.fetch_count, 1)
        }


class KnowledgeBase:
    """Persistent knowledge storage and retrieval"""
    
    def __init__(self, persistence_path: str = "/home/claude/knowledge_base.json"):
        self.persistence_path = persistence_path
        self.knowledge: Dict[str, Any] = {}
        self.insights: List[Dict[str, Any]] = []
        self.load()
    
    def store(self, key: str, value: Any, category: str = "general"):
        """Store knowledge"""
        if category not in self.knowledge:
            self.knowledge[category] = {}
        
        self.knowledge[category][key] = {
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'access_count': 0
        }
    
    def retrieve(self, key: str, category: str = "general") -> Optional[Any]:
        """Retrieve knowledge"""
        if category in self.knowledge and key in self.knowledge[category]:
            entry = self.knowledge[category][key]
            entry['access_count'] += 1
            return entry['value']
        return None
    
    def add_insight(self, insight: Dict[str, Any]):
        """Add learned insight"""
        insight['timestamp'] = datetime.now().isoformat()
        self.insights.append(insight)
        
        # Keep only recent insights
        if len(self.insights) > 1000:
            self.insights = self.insights[-1000:]
    
    def get_recent_insights(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent insights"""
        return self.insights[-n:]
    
    def save(self):
        """Save knowledge to disk"""
        try:
            with open(self.persistence_path, 'w') as f:
                json.dump({
                    'knowledge': self.knowledge,
                    'insights': self.insights
                }, f, indent=2, default=str)
            logger.info(f"Knowledge base saved to {self.persistence_path}")
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")
    
    def load(self):
        """Load knowledge from disk"""
        try:
            if os.path.exists(self.persistence_path):
                with open(self.persistence_path, 'r') as f:
                    data = json.load(f)
                    self.knowledge = data.get('knowledge', {})
                    self.insights = data.get('insights', [])
                logger.info(f"Knowledge base loaded from {self.persistence_path}")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")


class SelfEvolvingIntelligentSystem:
    """
    Universal Self-Evolving Intelligent System
    
    Features:
    - Continuous learning from data streams
    - Pattern recognition and mining
    - Self-writing code evolution
    - Always-on operation with auto-recovery
    - Knowledge persistence
    - Performance optimization
    """
    
    def __init__(
        self,
        system_id: str = "universal-ai-system",
        feature_dim: int = 50,
        learning_rate: float = 0.01,
        enable_code_evolution: bool = True,
        enable_always_on: bool = True,
        checkpoint_interval: int = 300
    ):
        self.system_id = system_id
        self.enable_code_evolution = enable_code_evolution
        
        logger.info(f"Initializing {system_id}...")
        
        # Core components
        self.learning_engine = ContinuousLearningEngine(
            feature_dim=feature_dim,
            learning_rate=learning_rate,
            pattern_mining=True,
            auto_adapt=True
        )
        
        self.knowledge_base = KnowledgeBase()
        
        # Code evolution (optional)
        self.code_evolver = None
        if enable_code_evolution:
            self.code_evolver = SelfEvolvingSystem(
                max_generations=10,
                population_size=20
            )
        
        # Always-on orchestration
        self.orchestrator = None
        if enable_always_on:
            self.orchestrator = AlwaysOnOrchestrator(
                checkpoint_interval=checkpoint_interval,
                auto_restart=True
            )
        
        # Data streams
        self.data_streams: Dict[str, DataStreamAdapter] = {}
        
        # Processing state
        self.running = False
        self.iteration = 0
        self.start_time = None
        
        # Performance metrics
        self.total_observations = 0
        self.total_predictions = 0
        self.total_patterns_found = 0
        
        # Decision making
        self.decision_history = []
        
        logger.info("System initialized successfully")
    
    def register_data_stream(self, stream_id: str, fetch_fn: Callable):
        """Register a data stream"""
        adapter = DataStreamAdapter(stream_id, fetch_fn)
        self.data_streams[stream_id] = adapter
        logger.info(f"Registered data stream: {stream_id}")
    
    def process_stream_data(self, stream_id: str) -> Dict[str, Any]:
        """Process data from a specific stream"""
        
        if stream_id not in self.data_streams:
            logger.error(f"Unknown stream: {stream_id}")
            return {}
        
        # Fetch data
        stream = self.data_streams[stream_id]
        data = stream.fetch()
        
        if not data:
            return {}
        
        # Extract outcome if present
        outcome = data.pop('outcome', None)
        feedback = data.pop('feedback', None)
        
        # Process through learning engine
        result = self.learning_engine.process_observation(
            data, 
            outcome=outcome,
            feedback=feedback
        )
        
        self.total_observations += 1
        self.total_predictions += 1
        
        # Store insight if significant
        if result.get('pattern_id'):
            self.total_patterns_found += 1
            self.knowledge_base.add_insight({
                'type': 'pattern_discovered',
                'stream_id': stream_id,
                'pattern_id': result['pattern_id'],
                'prediction': result['prediction']
            })
        
        # Store knowledge
        self.knowledge_base.store(
            key=f"{stream_id}_latest",
            value=result,
            category='stream_data'
        )
        
        return result
    
    def make_decision(
        self, 
        context: Dict[str, Any],
        decision_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Make intelligent decision based on context"""
        
        # Get prediction from learning engine
        prediction_result = self.learning_engine.process_observation(context)
        
        # Apply custom decision function if provided
        if decision_fn:
            decision = decision_fn(prediction_result, context)
        else:
            # Default: threshold-based decision
            decision = {
                'action': 'act' if prediction_result['prediction'] > 0.5 else 'wait',
                'confidence': abs(prediction_result['prediction'] - 0.5) * 2
            }
        
        # Record decision
        decision_record = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'prediction': prediction_result['prediction'],
            'decision': decision,
            'pattern_id': prediction_result.get('pattern_id')
        }
        
        self.decision_history.append(decision_record)
        
        # Store in knowledge base
        self.knowledge_base.add_insight({
            'type': 'decision_made',
            'decision': decision,
            'confidence': decision.get('confidence', 0)
        })
        
        return decision
    
    def evolve_component(
        self,
        component_code: str,
        test_cases: List[Dict[str, Any]],
        fitness_fn: Optional[Callable] = None
    ) -> str:
        """Evolve a code component for better performance"""
        
        if not self.code_evolver:
            logger.warning("Code evolution not enabled")
            return component_code
        
        logger.info("Starting code evolution...")
        
        best_variant = self.code_evolver.evolve_code(
            component_code,
            test_cases,
            fitness_fn
        )
        
        # Store evolved code
        self.knowledge_base.store(
            key='evolved_component',
            value={
                'code': best_variant.code,
                'score': best_variant.performance_score,
                'generation': best_variant.generation
            },
            category='evolved_code'
        )
        
        self.knowledge_base.add_insight({
            'type': 'code_evolved',
            'score': best_variant.performance_score,
            'generation': best_variant.generation
        })
        
        return best_variant.code
    
    def run_processing_loop(
        self,
        iterations: Optional[int] = None,
        interval: float = 1.0
    ):
        """Run main processing loop"""
        
        self.running = True
        self.start_time = datetime.now()
        self.iteration = 0
        
        logger.info("Starting processing loop...")
        logger.info(f"Registered streams: {list(self.data_streams.keys())}")
        
        try:
            while self.running:
                if iterations and self.iteration >= iterations:
                    break
                
                self.iteration += 1
                
                # Process all data streams
                for stream_id in self.data_streams.keys():
                    try:
                        result = self.process_stream_data(stream_id)
                        
                        # Log progress
                        if self.iteration % 10 == 0 and result:
                            logger.info(
                                f"Iteration {self.iteration} | "
                                f"Stream: {stream_id} | "
                                f"Prediction: {result.get('prediction', 0):.3f}"
                            )
                    except Exception as e:
                        logger.error(f"Error processing {stream_id}: {e}")
                
                # Periodic insights
                if self.iteration % 50 == 0:
                    insights = self.learning_engine.get_insights()
                    logger.info(
                        f"Insights | "
                        f"Accuracy: {insights['metrics']['accuracy']:.3f} | "
                        f"Patterns: {insights.get('total_patterns', 0)} | "
                        f"Samples: {insights['metrics']['samples_processed']}"
                    )
                
                # Periodic checkpoint
                if self.iteration % 100 == 0:
                    self.save_state()
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.shutdown()
    
    def save_state(self):
        """Save complete system state"""
        logger.info("Saving system state...")
        
        # Save learning engine
        self.learning_engine.save_state('/home/claude/learning_state.pkl')
        
        # Save knowledge base
        self.knowledge_base.save()
        
        # Save orchestrator state if enabled
        if self.orchestrator:
            self.orchestrator.save_state()
        
        logger.info("System state saved")
    
    def load_state(self):
        """Load complete system state"""
        logger.info("Loading system state...")
        
        try:
            # Load learning engine
            self.learning_engine.load_state('/home/claude/learning_state.pkl')
            
            # Load knowledge base
            self.knowledge_base.load()
            
            # Load orchestrator state if enabled
            if self.orchestrator:
                self.orchestrator.load_state()
            
            logger.info("System state loaded")
        except Exception as e:
            logger.warning(f"Could not fully load state: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        status = {
            'system_id': self.system_id,
            'running': self.running,
            'uptime_seconds': uptime,
            'iteration': self.iteration,
            'total_observations': self.total_observations,
            'total_predictions': self.total_predictions,
            'total_patterns': self.total_patterns_found,
            'learning_insights': self.learning_engine.get_insights(),
            'stream_stats': {
                stream_id: adapter.get_stats()
                for stream_id, adapter in self.data_streams.items()
            },
            'knowledge_base_size': len(self.knowledge_base.knowledge),
            'recent_insights': self.knowledge_base.get_recent_insights(5)
        }
        
        if self.orchestrator:
            status['orchestrator'] = self.orchestrator.get_status()
        
        return status
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down system...")
        
        self.running = False
        
        # Save final state
        self.save_state()
        
        # Stop orchestrator
        if self.orchestrator:
            self.orchestrator.stop_all()
        
        # Print summary
        status = self.get_system_status()
        
        logger.info("="*60)
        logger.info("SYSTEM SHUTDOWN SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Runtime: {status['uptime_seconds']:.0f}s")
        logger.info(f"Observations Processed: {status['total_observations']}")
        logger.info(f"Patterns Discovered: {status['total_patterns']}")
        logger.info(f"Learning Accuracy: {status['learning_insights']['metrics']['accuracy']:.3f}")
        logger.info("="*60)


def main():
    """Main entry point with example usage"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create system
    system = SelfEvolvingIntelligentSystem(
        system_id="demo-system",
        feature_dim=20,
        learning_rate=0.01,
        enable_code_evolution=True,
        enable_always_on=True
    )
    
    # Example data stream functions
    def sensor_stream_1():
        """Simulated sensor data"""
        import numpy as np
        return {
            'temperature': 20 + np.random.randn() * 5,
            'humidity': 50 + np.random.randn() * 10,
            'pressure': 1013 + np.random.randn() * 5,
            'outcome': np.random.randn()  # What we're trying to predict
        }
    
    def sensor_stream_2():
        """Another simulated sensor"""
        import numpy as np
        return {
            'voltage': 5.0 + np.random.randn() * 0.1,
            'current': 2.0 + np.random.randn() * 0.2,
            'active': np.random.choice([True, False]),
            'outcome': np.random.randn()
        }
    
    # Register streams
    system.register_data_stream('sensor_1', sensor_stream_1)
    system.register_data_stream('sensor_2', sensor_stream_2)
    
    # Load previous state if exists
    system.load_state()
    
    # Run!
    print("="*60)
    print("SELF-EVOLVING INTELLIGENT SYSTEM")
    print("="*60)
    print(f"System ID: {system.system_id}")
    print(f"Streams: {list(system.data_streams.keys())}")
    print(f"Code Evolution: {'Enabled' if system.code_evolver else 'Disabled'}")
    print(f"Always-On: {'Enabled' if system.orchestrator else 'Disabled'}")
    print("="*60)
    print("\nStarting... (Press Ctrl+C to stop)\n")
    
    system.run_processing_loop(iterations=1000, interval=0.5)


if __name__ == "__main__":
    main()
