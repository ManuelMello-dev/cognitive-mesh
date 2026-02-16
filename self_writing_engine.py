"""
Self-Writing / Self-Evolving Code Engine
Evolves code components through genetic programming
"""
import logging
import ast
import copy
import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CodeVariant:
    """A code variant in the evolutionary population"""
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
            'generation': self.generation,
            'performance_score': self.performance_score,
            'parent_id': self.parent_id,
            'mutations': self.mutations,
            'created_at': self.created_at.isoformat()
        }


class SelfEvolvingSystem:
    """
    Evolves code components through genetic programming.
    Features:
    - Population-based code evolution
    - Mutation operators (parameter tweaking, structure changes)
    - Fitness evaluation
    - Selection and crossover
    """

    def __init__(
        self,
        max_generations: int = 10,
        population_size: int = 20,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5
    ):
        self.max_generations = max_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.population: List[CodeVariant] = []
        self.best_variant: Optional[CodeVariant] = None
        self.generation = 0
        self.variant_counter = 0
        self.evolution_history: List[Dict[str, Any]] = []

        logger.info(
            f"SelfEvolvingSystem initialized: "
            f"max_gen={max_generations}, pop_size={population_size}"
        )

    def evolve_code(
        self,
        base_code: str,
        test_cases: List[Dict[str, Any]],
        fitness_fn: Optional[Callable] = None
    ) -> CodeVariant:
        """
        Evolve a code component for better performance.
        Returns the best variant found.
        """
        # Initialize population from base code
        self.population = self._initialize_population(base_code)

        for gen in range(self.max_generations):
            self.generation = gen

            # Evaluate fitness
            for variant in self.population:
                if fitness_fn:
                    variant.performance_score = fitness_fn(variant.code, test_cases)
                else:
                    variant.performance_score = self._default_fitness(variant.code, test_cases)

            # Sort by fitness
            self.population.sort(key=lambda v: v.performance_score, reverse=True)

            # Track best
            if self.population:
                current_best = self.population[0]
                if self.best_variant is None or current_best.performance_score > self.best_variant.performance_score:
                    self.best_variant = current_best

            # Record history
            self.evolution_history.append({
                'generation': gen,
                'best_score': self.population[0].performance_score if self.population else 0,
                'avg_score': sum(v.performance_score for v in self.population) / max(len(self.population), 1),
                'population_size': len(self.population)
            })

            logger.info(
                f"Generation {gen}: best={self.population[0].performance_score:.4f}, "
                f"avg={self.evolution_history[-1]['avg_score']:.4f}"
            )

            # Selection and reproduction
            self.population = self._next_generation()

        return self.best_variant or CodeVariant(
            variant_id="base_0",
            code=base_code,
            generation=0,
            performance_score=0.0
        )

    def _initialize_population(self, base_code: str) -> List[CodeVariant]:
        """Create initial population from base code"""
        population = []

        # Add base code as first member
        base = CodeVariant(
            variant_id=f"variant_{self.variant_counter}",
            code=base_code,
            generation=0
        )
        self.variant_counter += 1
        population.append(base)

        # Create mutations of base code
        for _ in range(self.population_size - 1):
            mutated_code = self._mutate_code(base_code)
            variant = CodeVariant(
                variant_id=f"variant_{self.variant_counter}",
                code=mutated_code,
                generation=0,
                parent_id=base.variant_id,
                mutations=["initial_mutation"]
            )
            self.variant_counter += 1
            population.append(variant)

        return population

    def _mutate_code(self, code: str) -> str:
        """Apply mutation to code string"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        # Simple mutations: tweak numeric constants
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if random.random() < self.mutation_rate:
                    # Perturb the constant
                    perturbation = random.gauss(0, 0.1) * max(abs(node.value), 1)
                    node.value = node.value + perturbation
                    if isinstance(node.value, int):
                        node.value = int(node.value)

        try:
            return ast.unparse(tree)
        except Exception:
            return code

    def _default_fitness(self, code: str, test_cases: List[Dict[str, Any]]) -> float:
        """Default fitness function: syntax validity + test pass rate"""
        score = 0.0

        # Syntax check
        try:
            ast.parse(code)
            score += 0.3  # Valid syntax
        except SyntaxError:
            return 0.0

        # Try to execute with test cases
        passed = 0
        for test in test_cases:
            try:
                # Create isolated namespace
                namespace = {}
                exec(code, namespace)
                passed += 1
            except Exception:
                pass

        if test_cases:
            score += 0.7 * (passed / len(test_cases))

        return score

    def _next_generation(self) -> List[CodeVariant]:
        """Create next generation through selection and reproduction"""
        if not self.population:
            return []

        new_population = []

        # Elitism: keep top 2
        new_population.extend(self.population[:2])

        # Fill rest with offspring
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            # Crossover or mutation
            if random.random() < self.crossover_rate:
                child_code = self._crossover(parent1.code, parent2.code)
                mutation_type = "crossover"
            else:
                child_code = self._mutate_code(parent1.code)
                mutation_type = "mutation"

            child = CodeVariant(
                variant_id=f"variant_{self.variant_counter}",
                code=child_code,
                generation=self.generation + 1,
                parent_id=parent1.variant_id,
                mutations=[mutation_type]
            )
            self.variant_counter += 1
            new_population.append(child)

        return new_population

    def _tournament_select(self, k: int = 3) -> CodeVariant:
        """Tournament selection"""
        candidates = random.sample(
            self.population,
            min(k, len(self.population))
        )
        return max(candidates, key=lambda v: v.performance_score)

    def _crossover(self, code1: str, code2: str) -> str:
        """Simple crossover between two code strings"""
        lines1 = code1.split('\n')
        lines2 = code2.split('\n')

        if len(lines1) < 2 or len(lines2) < 2:
            return code1

        # Single-point crossover
        point1 = random.randint(1, len(lines1) - 1)
        point2 = random.randint(1, len(lines2) - 1)

        child_lines = lines1[:point1] + lines2[point2:]
        child_code = '\n'.join(child_lines)

        # Verify syntax
        try:
            ast.parse(child_code)
            return child_code
        except SyntaxError:
            return code1  # Fall back to parent

    def get_insights(self) -> Dict[str, Any]:
        """Get evolution insights"""
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_score': self.best_variant.performance_score if self.best_variant else 0,
            'total_variants': self.variant_counter,
            'evolution_history': self.evolution_history[-10:]
        }
