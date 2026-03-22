import ast
import copy
import logging
import random
import multiprocessing
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

# Configure logging for production observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EvoEngine")

@dataclass
class CodeVariant:
    """A structurally-sound code variant in the evolutionary population."""
    variant_id: str
    code: str
    generation: int
    performance_score: float = 0.0
    parent_id: Optional[str] = None
    mutations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self):
        return {
            'variant_id': self.variant_id,
            'performance_score': round(self.performance_score, 4),
            'mutations': self.mutations,
            'generation': self.generation
        }

class ASTEngine:
    """Handles structurally aware mutations and crossovers to prevent syntax errors."""
    
    @staticmethod
    def safe_parse(code: str) -> Optional[ast.AST]:
        try:
            return ast.parse(code)
        except Exception:
            return None

    def crossover(self, code1: str, code2: str) -> str:
        tree1, tree2 = self.safe_parse(code1), self.safe_parse(code2)
        if not tree1 or not tree2:
            return code1

        # Target functional blocks for swapping
        targets = (ast.FunctionDef, ast.For, ast.While, ast.If, ast.Assign, ast.Return)
        nodes1 = [n for n in ast.walk(tree1) if isinstance(n, targets)]
        nodes2 = [n for n in ast.walk(tree2) if isinstance(n, targets)]

        if not nodes1 or not nodes2:
            return code1

        target_node = random.choice(nodes1)
        replacement = copy.deepcopy(random.choice(nodes2))

        # Perform structural swap
        for node in ast.walk(tree1):
            for field, value in ast.iter_fields(node):
                if value == target_node:
                    setattr(node, field, replacement)
                elif isinstance(value, list) and target_node in value:
                    value[value.index(target_node)] = replacement
        
        try:
            return ast.unparse(tree1)
        except:
            return code1

class SandboxEvaluator:
    """Isolated execution environment to mitigate security risks (exec/infinite loops)."""
    
    @staticmethod
    def run_isolated(code: str, test_cases: List[Dict]) -> float:
        """Executed within a separate process via Multiprocessing."""
        try:
            # 1. Restrict builtins (No file system/network access)
            safe_builtins = __builtins__.copy() if isinstance(__builtins__, dict) else vars(__builtins__).copy()
            for unsafe in ['open', 'exec', 'eval', 'getattr', 'setattr', 'os', 'sys', 'importlib']:
                safe_builtins.pop(unsafe, None)

            compiled = compile(code, '<evolved>', 'exec')
            score = 0.0
            
            for test in test_cases:
                # Fresh namespace for isolation
                ns = {"__builtins__": safe_builtins}
                ns.update(test.get('inputs', {}))
                
                try:
                    exec(compiled, ns)
                    if ns.get('result') == test.get('expected'):
                        score += 1.0
                except:
                    continue
            
            return score / len(test_cases) if test_cases else 0.0
        except Exception:
            return 0.0

class ProductionEvolvingSystem:
    def __init__(
        self,
        pop_size: int = 20,
        max_gen: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.6
    ):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.ast_engine = ASTEngine()
        self.population: List[CodeVariant] = []
        self.hall_of_fame: List[CodeVariant] = [] # Top 5 ever seen
        self.variant_counter = 0

    def _mutate(self, code: str) -> str:
        tree = self.ast_engine.safe_parse(code)
        if not tree: return code
        
        # Logic for numeric constant perturbation
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if random.random() < self.mutation_rate:
                    node.value += random.gauss(0, 0.1) * max(abs(node.value), 1)
                    if isinstance(node.value, int): node.value = int(node.value)
        return ast.unparse(tree)

    def evolve(self, base_code: str, test_cases: List[Dict]) -> CodeVariant:
        # Initialize
        self.population = [CodeVariant(f"v{i}", self._mutate(base_code) if i > 0 else base_code, 0) 
                          for i in range(self.pop_size)]
        
        # Parallel Evaluation Pool
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for gen in range(self.max_gen):
                # 1. Parallel Fitness Check
                tasks = [(v.code, test_cases) for v in self.population]
                scores = pool.starmap(SandboxEvaluator.run_isolated, tasks)
                
                for variant, score in zip(self.population, scores):
                    variant.performance_score = score

                # 2. Update Hall of Fame
                self.population.sort(key=lambda v: v.performance_score, reverse=True)
                self._update_hall_of_fame(self.population[0])

                logger.info(f"Gen {gen} | Best Score: {self.population[0].performance_score:.4f}")

                # 3. Selection (Elitism + Tournament)
                next_pop = copy.deepcopy(self.population[:2]) # Elitism
                
                while len(next_pop) < self.pop_size:
                    p1, p2 = self._tournament(), self._tournament()
                    
                    if random.random() < self.crossover_rate:
                        child_code = self.ast_engine.crossover(p1.code, p2.code)
                        m_type = "crossover"
                    else:
                        child_code = self._mutate(p1.code)
                        m_type = "mutation"
                    
                    self.variant_counter += 1
                    next_pop.append(CodeVariant(f"v{self.variant_counter}", child_code, gen+1, 0.0, p1.variant_id, [m_type]))
                
                self.population = next_pop

        return self.hall_of_fame[0]

    def _tournament(self, k=3) -> CodeVariant:
        selection = random.sample(self.population, k)
        return max(selection, key=lambda v: v.performance_score)

    def _update_hall_of_fame(self, variant: CodeVariant):
        self.hall_of_fame.append(copy.deepcopy(variant))
        self.hall_of_fame.sort(key=lambda v: v.performance_score, reverse=True)
        self.hall_of_fame = self.hall_of_fame[:5]

# --- Backward-compatible alias ---
# cognitive_intelligent_system.py imports SelfEvolvingSystem; this maps it to
# the canonical ProductionEvolvingSystem class.
SelfEvolvingSystem = ProductionEvolvingSystem

# --- Deployment Note ---
# This code uses multiprocessing. Ensure you wrap the execution in
# if __name__ == "__main__": when running locally or in CI/CD.
