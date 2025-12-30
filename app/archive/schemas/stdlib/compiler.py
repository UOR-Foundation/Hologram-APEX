"""
Python to JSON Schema Compiler

Compiles Python functions to JSON schemas for Atlas kernel generation.
Based on historical frontends/atlas_py/compiler.py
"""

import ast
import json
import inspect
from typing import Any, Callable, Dict, List, Optional
from functools import wraps


class AtlasCompiler:
    """Compiles Python AST to Atlas JSON schema"""

    # Built-in intrinsic functions (compiler-implemented, not user-defined)
    INTRINSICS = {
        'get_global_id', 'atomic_add_f32', 'atomic_add_u32', 'atomic_add_i32',
        'atomic_add', 'atomic_min', 'expf', 'logf', 'sinf', 'cosf', 'sqrtf',
        'sqrt', 'rsqrt', 'exp',
        # Phase 2B: Higher-order primitives
        'parallel_map_unary', 'parallel_map_binary', 'parallel_reduce',
        'parallel_scan', 'parallel_gather', 'parallel_scatter',
    }

    def __init__(self):
        self.declared_vars = set()  # Track declared variables for assignment vs declaration
        self.inline_functions = {}  # Registry of inline functions: {name: FunctionDef}

    def compile_function(self, func: Callable, module_source: Optional[str] = None) -> Dict[str, Any]:
        """
        Compile a Python function to JSON schema with inline function support

        Args:
            func: The function to compile
            module_source: Optional module source code (to capture inline functions)
        """
        # Reset state for each function compilation
        self.declared_vars = set()
        self.inline_functions = {}

        # Get source code - use provided module_source or try module inspection
        if module_source:
            tree = ast.parse(module_source)
        else:
            try:
                # Get the module source to capture inline function definitions
                module = inspect.getmodule(func)
                if module:
                    module_source = inspect.getsource(module)
                    tree = ast.parse(module_source)
                else:
                    # Fallback to function source only
                    source = inspect.getsource(func)
                    tree = ast.parse(source)
            except (OSError, TypeError):
                # Fallback to function source only
                source = inspect.getsource(func)
                tree = ast.parse(source)

        # **PHASE 1**: Collect all @inline function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if self._has_inline_decorator(node):
                    self.inline_functions[node.name] = node

        # **PHASE 2**: Find and compile the main kernel function
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
                func_def = node
                break

        if not func_def:
            raise ValueError(f"Could not find function {func.__name__}")

        # Extract parameters and mark them as declared
        params = self._compile_parameters(func_def)
        for param in params:
            self.declared_vars.add(param["name"])

        # Compile function body (function calls will be inlined automatically)
        body = self._compile_body(func_def.body)

        return {
            "version": "1.0",
            "kernel": {
                "name": func.__name__,
                "params": params,
                "body": body
            }
        }

    def _compile_parameters(self, func_def: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract and compile function parameters"""
        params = []

        for arg in func_def.args.args:
            param_name = arg.arg
            param_type = self._parse_type_annotation(arg.annotation)

            params.append({
                "name": param_name,
                "type": param_type
            })

        return params

    def _parse_type_annotation(self, annotation) -> Dict[str, Any]:
        """Parse Python type annotation to Atlas type"""
        if annotation is None:
            raise ValueError("All kernel parameters must have type annotations")

        # DeviceArray[type]
        if isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                if annotation.value.id == "DeviceArray":
                    element_type = self._parse_scalar_type(annotation.slice)
                    return {
                        "kind": "device_array",
                        "element_type": element_type
                    }

        # Scalar types
        if isinstance(annotation, ast.Name):
            return self._parse_scalar_type(annotation)

        raise ValueError(f"Unsupported type annotation: {ast.dump(annotation)}")

    def _parse_scalar_type(self, node) -> Dict[str, Any]:
        """Parse scalar type"""
        if isinstance(node, ast.Name):
            type_map = {
                "int": "i32",
                "float": "f32",
                "bool": "bool",
                "u8": "u8",
                "u16": "u16",
                "u32": "u32",
                "u64": "u64",
                "i8": "i8",
                "i16": "i16",
                "i32": "i32",
                "i64": "i64",
                "f32": "f32",
                "f64": "f64",
                "usize": "usize",
            }

            scalar_type = type_map.get(node.id)
            if scalar_type:
                return {
                    "kind": "scalar",
                    "type": scalar_type
                }

        raise ValueError(f"Unsupported scalar type: {ast.dump(node)}")

    def _compile_body(self, statements: List[ast.stmt]) -> List[Dict[str, Any]]:
        """Compile function body statements"""
        result = []

        for stmt in statements:
            compiled = self._compile_statement(stmt)
            if compiled:
                result.append(compiled)

        return result

    def _compile_statement(self, stmt: ast.stmt) -> Optional[Dict[str, Any]]:
        """Compile a single statement"""
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) == 1:
                target = stmt.targets[0]
                # Variable declaration or reassignment
                if isinstance(target, ast.Name):
                    var_name = target.id
                    # Check if this variable has already been declared
                    if var_name in self.declared_vars:
                        # Reassignment
                        return {
                            "type": "assign",
                            "target": {"type": "var", "name": var_name},
                            "value": self._compile_expression(stmt.value)
                        }
                    else:
                        # Declaration
                        self.declared_vars.add(var_name)
                        return {
                            "type": "let",
                            "name": var_name,
                            "value": self._compile_expression(stmt.value)
                        }
                # Array assignment
                elif isinstance(target, ast.Subscript):
                    return {
                        "type": "assign",
                        "target": self._compile_expression(target),
                        "value": self._compile_expression(stmt.value)
                    }

        elif isinstance(stmt, ast.AugAssign):
            # Augmented assignment: c[idx] += value
            return {
                "type": "assign",
                "target": self._compile_expression(stmt.target),
                "value": {
                    "type": "binary_op",
                    "op": self._compile_aug_op(stmt.op),
                    "left": self._compile_expression(stmt.target),
                    "right": self._compile_expression(stmt.value)
                }
            }

        elif isinstance(stmt, ast.Expr):
            # Check if this is an intrinsic call (like parallel_map_unary)
            if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name):
                if stmt.value.func.id in self.INTRINSICS:
                    # Compile intrinsic call as a statement
                    return self._compile_intrinsic_call(stmt.value)
            return None  # Skip other bare expressions

        elif isinstance(stmt, ast.If):
            return {
                "type": "if",
                "condition": self._compile_expression(stmt.test),
                "then_body": self._compile_body(stmt.body),
                "else_body": self._compile_body(stmt.orelse) if stmt.orelse else None
            }

        elif isinstance(stmt, ast.Return):
            return {
                "type": "return",
                "value": self._compile_expression(stmt.value) if stmt.value else None
            }

        elif isinstance(stmt, ast.For):
            # Handle range() loops
            if isinstance(stmt.iter, ast.Call) and isinstance(stmt.iter.func, ast.Name):
                if stmt.iter.func.id == "range":
                    # Extract range arguments
                    if len(stmt.iter.args) == 1:
                        start = {"type": "literal", "value": 0}
                        stop = self._compile_expression(stmt.iter.args[0])
                        step = {"type": "literal", "value": 1}
                    elif len(stmt.iter.args) == 2:
                        start = self._compile_expression(stmt.iter.args[0])
                        stop = self._compile_expression(stmt.iter.args[1])
                        step = {"type": "literal", "value": 1}
                    elif len(stmt.iter.args) == 3:
                        start = self._compile_expression(stmt.iter.args[0])
                        stop = self._compile_expression(stmt.iter.args[1])
                        step = self._compile_expression(stmt.iter.args[2])
                    else:
                        raise ValueError("Invalid range() call")

                    var_name = stmt.target.id if isinstance(stmt.target, ast.Name) else None
                    if not var_name:
                        raise ValueError("For loop target must be a simple variable")

                    self.declared_vars.add(var_name)

                    return {
                        "type": "for",
                        "var": var_name,
                        "start": start,
                        "stop": stop,
                        "step": step,
                        "body": self._compile_body(stmt.body)
                    }

        return None

    def _compile_expression(self, expr: ast.expr) -> Dict[str, Any]:
        """Compile an expression"""
        if isinstance(expr, ast.Name):
            return {"type": "var", "name": expr.id}

        elif isinstance(expr, ast.Constant):
            return {"type": "literal", "value": expr.value}

        elif isinstance(expr, ast.BinOp):
            return {
                "type": "binary_op",
                "op": self._compile_binop(expr.op),
                "left": self._compile_expression(expr.left),
                "right": self._compile_expression(expr.right)
            }

        elif isinstance(expr, ast.Compare):
            # Handle single comparison
            if len(expr.ops) == 1 and len(expr.comparators) == 1:
                return {
                    "type": "binary_op",
                    "op": self._compile_cmpop(expr.ops[0]),
                    "left": self._compile_expression(expr.left),
                    "right": self._compile_expression(expr.comparators[0])
                }

        elif isinstance(expr, ast.Subscript):
            return {
                "type": "index",
                "array": self._compile_expression(expr.value),
                "index": self._compile_expression(expr.slice)
            }

        elif isinstance(expr, ast.UnaryOp):
            return {
                "type": "unary_op",
                "op": self._compile_unaryop(expr.op),
                "operand": self._compile_expression(expr.operand)
            }

        elif isinstance(expr, ast.Call):
            # Check if this is a higher-order primitive (intrinsic)
            if isinstance(expr.func, ast.Name) and expr.func.id in self.INTRINSICS:
                return self._compile_intrinsic_call(expr)
            # Use inline function handler (handles both inline and regular calls)
            return self._inline_function_call(expr)

        elif isinstance(expr, ast.IfExp):
            # Ternary expression: a if condition else b
            return {
                "type": "if_expr",
                "condition": self._compile_expression(expr.test),
                "then": self._compile_expression(expr.body),
                "else": self._compile_expression(expr.orelse)
            }

        raise ValueError(f"Unsupported expression: {ast.dump(expr)}")

    def _compile_binop(self, op: ast.operator) -> str:
        """Compile binary operator"""
        op_map = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.Div: "div",
            ast.Mod: "mod",
        }
        return op_map.get(type(op), "unknown")

    def _compile_cmpop(self, op: ast.cmpop) -> str:
        """Compile comparison operator"""
        op_map = {
            ast.Lt: "lt",
            ast.LtE: "le",
            ast.Gt: "gt",
            ast.GtE: "ge",
            ast.Eq: "eq",
            ast.NotEq: "ne",
        }
        return op_map.get(type(op), "unknown")

    def _compile_aug_op(self, op: ast.operator) -> str:
        """Compile augmented assignment operator"""
        return self._compile_binop(op)

    def _compile_unaryop(self, op: ast.unaryop) -> str:
        """Compile unary operator"""
        op_map = {
            ast.UAdd: "pos",    # Unary +
            ast.USub: "neg",    # Unary -
            ast.Not: "not",     # Logical not
            ast.Invert: "inv",  # Bitwise invert ~
        }
        return op_map.get(type(op), "unknown")

    def _has_inline_decorator(self, func_def: ast.FunctionDef) -> bool:
        """Check if a function has the @inline decorator"""
        for decorator in func_def.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "inline":
                return True
        return False

    def _compile_intrinsic_call(self, call: ast.Call) -> Dict[str, Any]:
        """
        Compile calls to built-in intrinsic functions (higher-order primitives).

        Intrinsics are special functions implemented by the backend, not user-defined.
        They generate specialized JSON structures for backend optimization.
        """
        func_name = call.func.id

        # parallel_map_unary(input, output, operation, n)
        if func_name == 'parallel_map_unary':
            if len(call.args) != 4:
                raise ValueError(f"parallel_map_unary requires 4 arguments, got {len(call.args)}")

            # Extract operation name (must be a string literal)
            operation_arg = call.args[2]
            if not isinstance(operation_arg, ast.Constant) or not isinstance(operation_arg.value, str):
                raise ValueError("parallel_map_unary operation must be a string literal")

            return {
                "type": "parallel_map_unary",
                "input": self._compile_expression(call.args[0]),
                "output": self._compile_expression(call.args[1]),
                "operation": operation_arg.value,
                "count": self._compile_expression(call.args[3])
            }

        # parallel_map_binary(input_a, input_b, output, operation, n)
        if func_name == 'parallel_map_binary':
            if len(call.args) != 5:
                raise ValueError(f"parallel_map_binary requires 5 arguments, got {len(call.args)}")

            # Extract operation name (must be a string literal)
            operation_arg = call.args[3]
            if not isinstance(operation_arg, ast.Constant) or not isinstance(operation_arg.value, str):
                raise ValueError("parallel_map_binary operation must be a string literal")

            return {
                "type": "parallel_map_binary",
                "input_a": self._compile_expression(call.args[0]),
                "input_b": self._compile_expression(call.args[1]),
                "output": self._compile_expression(call.args[2]),
                "operation": operation_arg.value,
                "count": self._compile_expression(call.args[4])
            }

        # parallel_reduce(input, output, operation, n)
        if func_name == 'parallel_reduce':
            if len(call.args) != 4:
                raise ValueError(f"parallel_reduce requires 4 arguments, got {len(call.args)}")

            # Extract operation name (must be a string literal)
            operation_arg = call.args[2]
            if not isinstance(operation_arg, ast.Constant) or not isinstance(operation_arg.value, str):
                raise ValueError("parallel_reduce operation must be a string literal")

            return {
                "type": "parallel_reduce",
                "input": self._compile_expression(call.args[0]),
                "output": self._compile_expression(call.args[1]),
                "operation": operation_arg.value,
                "count": self._compile_expression(call.args[3])
            }

        # For other intrinsics (exp, sqrt, get_global_id, etc.), treat as regular call
        return {
            "type": "call",
            "function": func_name,
            "args": [self._compile_expression(arg) for arg in call.args]
        }

    def _inline_function_call(self, call: ast.Call) -> Dict[str, Any]:
        """
        Replace a function call with the inlined function body.

        Returns the compiled expression resulting from the inlined function.
        """
        if not isinstance(call.func, ast.Name):
            # Not a simple function call, can't inline
            return {
                "type": "call",
                "function": ast.dump(call.func),
                "args": [self._compile_expression(arg) for arg in call.args]
            }

        func_name = call.func.id

        # Check if this is an inline function
        if func_name not in self.inline_functions:
            # Regular function call (e.g., exp, sqrt, etc.)
            return {
                "type": "call",
                "function": func_name,
                "args": [self._compile_expression(arg) for arg in call.args]
            }

        func_def = self.inline_functions[func_name]

        # Check recursion
        if self._contains_call_to(func_def.body, func_name):
            raise ValueError(f"Recursive inline function not allowed: {func_name}")

        # Build parameter mapping
        param_map = {}
        for param, arg in zip(func_def.args.args, call.args):
            param_map[param.arg] = arg

        # Inline the function body
        # Filter out docstrings (Expr nodes with Constant values)
        body_without_docstrings = [
            stmt for stmt in func_def.body
            if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Constant, ast.Str)))
        ]

        if len(body_without_docstrings) == 1 and isinstance(body_without_docstrings[0], ast.Return):
            # Simple case: single return statement
            return_expr = body_without_docstrings[0].value
            inlined_expr = self._substitute_parameters_in_expr(return_expr, param_map)
            return self._compile_expression(inlined_expr)
        else:
            raise ValueError(f"Inline function {func_name} must contain only a return statement (excluding docstring)")

    def _substitute_parameters_in_expr(self, expr: ast.expr, param_map: Dict[str, ast.expr]) -> ast.expr:
        """Substitute parameter references with argument expressions in an AST expression"""
        if isinstance(expr, ast.Name):
            # Replace parameter with argument
            if expr.id in param_map:
                return param_map[expr.id]
            return expr

        elif isinstance(expr, ast.BinOp):
            return ast.BinOp(
                left=self._substitute_parameters_in_expr(expr.left, param_map),
                op=expr.op,
                right=self._substitute_parameters_in_expr(expr.right, param_map)
            )

        elif isinstance(expr, ast.UnaryOp):
            return ast.UnaryOp(
                op=expr.op,
                operand=self._substitute_parameters_in_expr(expr.operand, param_map)
            )

        elif isinstance(expr, ast.Call):
            return ast.Call(
                func=expr.func,
                args=[self._substitute_parameters_in_expr(arg, param_map) for arg in expr.args],
                keywords=expr.keywords
            )

        elif isinstance(expr, ast.Compare):
            return ast.Compare(
                left=self._substitute_parameters_in_expr(expr.left, param_map),
                ops=expr.ops,
                comparators=[self._substitute_parameters_in_expr(comp, param_map) for comp in expr.comparators]
            )

        elif isinstance(expr, ast.IfExp):
            return ast.IfExp(
                test=self._substitute_parameters_in_expr(expr.test, param_map),
                body=self._substitute_parameters_in_expr(expr.body, param_map),
                orelse=self._substitute_parameters_in_expr(expr.orelse, param_map)
            )

        elif isinstance(expr, ast.Subscript):
            return ast.Subscript(
                value=self._substitute_parameters_in_expr(expr.value, param_map),
                slice=self._substitute_parameters_in_expr(expr.slice, param_map),
                ctx=expr.ctx
            )

        # For constants and other literals, return as-is
        return expr

    def _contains_call_to(self, body: List[ast.stmt], func_name: str) -> bool:
        """Check if function body contains a call to the specified function (recursion detection)"""
        for stmt in body:
            if self._stmt_contains_call_to(stmt, func_name):
                return True
        return False

    def _stmt_contains_call_to(self, stmt: ast.stmt, func_name: str) -> bool:
        """Check if a statement contains a call to the specified function"""
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == func_name:
                    return True
        return False


# Global compiler instance
_compiler = AtlasCompiler()


def atlas_kernel(func: Callable) -> Callable:
    """Decorator to mark function as Atlas kernel"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        raise RuntimeError(
            f"Kernel {func.__name__} cannot be called directly. "
            "Use compile_to_json() to compile it first."
        )
    
    wrapper._atlas_kernel = func
    return wrapper


def compile_to_json(func: Callable, output_path: Optional[str] = None) -> str:
    """Compile a kernel function to JSON schema"""
    # Get original function if wrapped
    original_func = getattr(func, '_atlas_kernel', func)
    
    # Compile to JSON
    schema = _compiler.compile_function(original_func)
    json_str = json.dumps(schema, indent=2)
    
    # Write to file if requested
    if output_path:
        with open(output_path, 'w') as f:
            f.write(json_str)
    
    return json_str
