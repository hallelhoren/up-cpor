import unittest
import sys
from typing import List, Dict
import unified_planning as up
from unified_planning.shortcuts import Fluent, BoolType, And, Or, Not, Equals, Implies

# ---------------------------------------------------------------------------
# Dynamic Production Import
# ---------------------------------------------------------------------------
# This forces the test to use your ACTUAL live code, not a mock.
# Adjust the import path if your file is named 'converter.py' instead of 'problem_grounder.py'
try:
    from up_cpor.problem_grounder import UpCporConverter, OP_AND, OP_OR, OP_NOT, OP_EQUALS, OP_TRUE, OP_FALSE
except ImportError:
    try:
        from up_cpor.converter import UpCporConverter, OP_AND, OP_OR, OP_NOT, OP_EQUALS, OP_TRUE, OP_FALSE
    except ImportError as e:
        print(f"CRITICAL FIX REQUIRED: Cannot import your production file. {e}")
        sys.exit(1)


class TestGrounderAdversarial(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Setup UP Environment
        up.shortcuts.get_environment()
        
        # Define base fluents
        cls.f_A = Fluent('A', BoolType())
        cls.f_B = Fluent('B', BoolType())
        cls.f_C = Fluent('C', BoolType())
        cls.f_Missing = Fluent('Missing', BoolType())
        
        # Instantiate your ACTUAL production converter
        cls.compiler = UpCporConverter()
        
        # Manually inject the registry to bypass the full problem initialization for unit testing
        cls.compiler.fluent_to_id = {
            "A": 0,
            "B": 1,
            "C": 2
        }

    def print_trace(self, name: str, expected, actual):
        print(f"\n--- {name} ---")
        print(f"EXPECTED: {expected}")
        print(f"ACTUAL:   {actual}")
        if expected != actual:
            print(">>> FAILURE: Stack mismatch detected.")

    # -------------------------------------------------------------------------
    # TIER 1: Standard / Common Cases
    # -------------------------------------------------------------------------
    def test_tier1_simple_conjunction(self):
        node = And(self.f_A, self.f_B)
        actual = self.compiler._compile_to_rpn(node)
        expected = [0, 1, OP_AND]
        
        self.print_trace("T1: Simple Conjunction AND(A, B)", expected, actual)
        self.assertEqual(actual, expected)

    def test_tier1_n_ary_disjunction(self):
        node = Or(self.f_A, self.f_B, self.f_C)
        actual = self.compiler._compile_to_rpn(node)
        expected = [0, 1, 2, OP_OR, OP_OR]
        
        self.print_trace("T1: N-Ary Disjunction OR(A, B, C)", expected, actual)
        self.assertEqual(actual, expected)

    # -------------------------------------------------------------------------
    # TIER 2: Complex / Deep Nesting
    # -------------------------------------------------------------------------
    def test_tier2_deep_nesting(self):
        node = And(self.f_A, Not(Or(self.f_B, self.f_C)))
        actual = self.compiler._compile_to_rpn(node)
        expected = [0, 1, 2, OP_OR, OP_NOT, OP_AND]
        
        self.print_trace("T2: Deep Nesting AND(A, NOT(OR(B, C)))", expected, actual)
        self.assertEqual(actual, expected)

    # -------------------------------------------------------------------------
    # TIER 3: Boundary / Edge Cases (The Breakers)
    # -------------------------------------------------------------------------
    def test_tier3_single_argument_collapse(self):
        node = And(self.f_A)
        actual = self.compiler._compile_to_rpn(node)
        expected = [0]
        
        self.print_trace("T3: 1-Arg Operator AND(A)", expected, actual)
        self.assertEqual(actual, expected)

    def test_tier3_missing_fluent_stack_corruption(self):
        node = And(self.f_A, self.f_Missing)
        
        # The test now asserts that the architectural protection (ValueError) triggers successfully.
        with self.assertRaises(ValueError) as context:
            self.compiler._compile_to_rpn(node)
            
        print(f"\n--- T3: Missing Fluent Drop AND(A, Missing) ---")
        print(f"EXPECTED EXCEPTION: ValueError")
        print(f"ACTUAL EXCEPTION:   {type(context.exception).__name__} - {str(context.exception)}")

    def test_tier3_unhandled_operators(self):
        node = Implies(self.f_A, self.f_B)
        actual = self.compiler._compile_to_rpn(node)
        
        # IMPLIES(A, B) translates mathematically to OR(NOT(A), B)
        # Expected Stack: [A, NOT, B, OR] -> [0, -3, 1, -2]
        expected = [0, OP_NOT, 1, OP_OR]
        
        self.print_trace("T3: Handled Node IMPLIES(A, B)", expected, actual)
        self.assertEqual(actual, expected, "Implies node failed to map to NOT/OR arity sequence.")


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=0)