import os
import sys
import re
import ast
import importlib.util
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to path to import debug_logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_bot.utils.debug_logger import DebugLogger

class DebugIntegrator:
    """
    Utility to help integrate debug logging into specific parts of the codebase.
    """
    
    def __init__(self, log_name="integrator"):
        """
        Initialize the debug integrator.
        
        Args:
            log_name (str): Base name for the log file
        """
        self.logger = DebugLogger(log_name=log_name)
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a Python file to find functions, classes, and entry points
        for adding debug logging.
        
        Args:
            file_path (str): Path to the Python file
            
        Returns:
            Dict: Analysis results including functions, classes, etc.
        """
        if not os.path.exists(file_path) or not file_path.endswith('.py'):
            self.logger.logger.error(f"Invalid Python file: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r') as f:
                source = f.read()
            
            # Parse the AST
            tree = ast.parse(source)
            
            # Extract information
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                # Find functions
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'has_return': any(isinstance(n, ast.Return) for n in ast.walk(node))
                    })
                
                # Find classes
                elif isinstance(node, ast.ClassDef):
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append({
                                'name': item.name,
                                'line': item.lineno,
                                'args': [arg.arg for arg in item.args.args if arg.arg != 'self'],
                                'has_return': any(isinstance(n, ast.Return) for n in ast.walk(item))
                            })
                    
                    classes.append({
                        'name': node.name,
                        'line': node.lineno,
                        'methods': methods
                    })
                
                # Find imports
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.append({
                                'module': name.name,
                                'alias': name.asname,
                                'line': node.lineno
                            })
                    else:  # ImportFrom
                        module = node.module or ''
                        for name in node.names:
                            imports.append({
                                'module': f"{module}.{name.name}" if module else name.name,
                                'alias': name.asname,
                                'line': node.lineno
                            })
            
            # Check if debug_logger is already imported
            has_debug_logger = any('debug_logger' in imp.get('module', '') for imp in imports)
            
            return {
                'file_path': file_path,
                'functions': functions,
                'classes': classes,
                'imports': imports,
                'has_debug_logger': has_debug_logger
            }
            
        except Exception as e:
            self.logger.logger.error(f"Error analyzing file {file_path}: {str(e)}")
            return {}
    
    def generate_logging_code(self, analysis: Dict[str, Any]) -> List[Tuple[int, str]]:
        """
        Generate logging code to insert into the file.
        
        Args:
            analysis (Dict): Analysis results from analyze_file
            
        Returns:
            List[Tuple[int, str]]: List of (line_number, code_to_insert)
        """
        if not analysis:
            return []
        
        inserts = []
        
        # Add import if needed
        if not analysis.get('has_debug_logger'):
            # Find the last import line
            import_lines = [imp['line'] for imp in analysis.get('imports', [])]
            insert_line = max(import_lines) + 1 if import_lines else 1
            
            inserts.append((
                insert_line,
                "from trading_bot.utils.debug_logger import log_input, log_output, log_error, log_variable, log_process\n"
            ))
        
        # Add logging to functions
        for func in analysis.get('functions', []):
            # Skip special methods
            if func['name'].startswith('__') and func['name'].endswith('__'):
                continue
            
            # Add input logging at the beginning of the function
            if func['args']:
                args_dict = "{" + ", ".join([f"'{arg}': {arg}" for arg in func['args']]) + "}"
                inserts.append((
                    func['line'] + 1,  # +1 to insert after the function definition
                    f"    log_input({args_dict}, source='{func['name']}')\n"
                ))
            
            # Add try-except if the function has a return
            if func['has_return']:
                inserts.append((
                    func['line'] + 1,  # +1 to insert after the function definition
                    "    try:\n"
                ))
                
                # We need to find all return statements and add logging
                # This is more complex and would require parsing the function body
                # For now, we'll add a comment as a placeholder
                inserts.append((
                    func['line'] + 2,
                    "    # TODO: Add log_output before each return statement\n"
                ))
                
                inserts.append((
                    func['line'] + 3,
                    "    except Exception as e:\n        log_error(e, context={'function': '" + func['name'] + "'})\n        raise\n"
                ))
        
        # Add logging to class methods
        for cls in analysis.get('classes', []):
            for method in cls['methods']:
                # Skip special methods
                if method['name'].startswith('__') and method['name'].endswith('__'):
                    continue
                
                # Add input logging at the beginning of the method
                if method['args']:
                    args_dict = "{" + ", ".join([f"'{arg}': {arg}" for arg in method['args']]) + "}"
                    inserts.append((
                        method['line'] + 1,  # +1 to insert after the method definition
                        f"        log_input({args_dict}, source='{cls['name']}.{method['name']}')\n"
                    ))
                
                # Add try-except if the method has a return
                if method['has_return']:
                    inserts.append((
                        method['line'] + 1,  # +1 to insert after the method definition
                        "        try:\n"
                    ))
                    
                    # We need to find all return statements and add logging
                    # This is more complex and would require parsing the method body
                    # For now, we'll add a comment as a placeholder
                    inserts.append((
                        method['line'] + 2,
                        "        # TODO: Add log_output before each return statement\n"
                    ))
                    
                    inserts.append((
                        method['line'] + 3,
                        "        except Exception as e:\n            log_error(e, context={'method': '" + cls['name'] + "." + method['name'] + "'})\n            raise\n"
                    ))
        
        return sorted(inserts, key=lambda x: x[0])
    
    def apply_logging(self, file_path: str, dry_run: bool = True) -> str:
        """
        Apply logging code to a file.
        
        Args:
            file_path (str): Path to the Python file
            dry_run (bool): If True, don't actually modify the file
            
        Returns:
            str: Modified code or empty string if dry_run is False
        """
        analysis = self.analyze_file(file_path)
        if not analysis:
            return ""
        
        inserts = self.generate_logging_code(analysis)
        if not inserts:
            return ""
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Apply inserts in reverse order to avoid changing line numbers
            for line_num, code in sorted(inserts, key=lambda x: x[0], reverse=True):
                lines.insert(line_num, code)
            
            modified_code = ''.join(lines)
            
            if not dry_run:
                with open(file_path, 'w') as f:
                    f.write(modified_code)
                self.logger.logger.info(f"Applied logging to {file_path}")
            
            return modified_code
            
        except Exception as e:
            self.logger.logger.error(f"Error applying logging to {file_path}: {str(e)}")
            return ""
    
    def apply_logging_to_directory(self, directory: str, exclude_patterns: List[str] = None, dry_run: bool = True) -> Dict[str, str]:
        """
        Apply logging to all Python files in a directory.
        
        Args:
            directory (str): Directory to process
            exclude_patterns (List[str]): List of regex patterns to exclude
            dry_run (bool): If True, don't actually modify the files
            
        Returns:
            Dict[str, str]: Dictionary of {file_path: modified_code}
        """
        if not os.path.isdir(directory):
            self.logger.logger.error(f"Invalid directory: {directory}")
            return {}
        
        exclude_patterns = exclude_patterns or []
        exclude_regex = [re.compile(pattern) for pattern in exclude_patterns]
        
        results = {}
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    # Check if file should be excluded
                    if any(pattern.search(file_path) for pattern in exclude_regex):
                        self.logger.logger.info(f"Skipping excluded file: {file_path}")
                        continue
                    
                    modified_code = self.apply_logging(file_path, dry_run)
                    if modified_code:
                        results[file_path] = modified_code
        
        return results
    
    def analyze_function_calls(self, file_path: str, function_name: str) -> List[Dict[str, Any]]:
        """
        Find all calls to a specific function in a file.
        
        Args:
            file_path (str): Path to the Python file
            function_name (str): Name of the function to find calls for
            
        Returns:
            List[Dict]: List of function call information
        """
        if not os.path.exists(file_path) or not file_path.endswith('.py'):
            self.logger.logger.error(f"Invalid Python file: {file_path}")
            return []
        
        try:
            with open(file_path, 'r') as f:
                source = f.read()
            
            # Parse the AST
            tree = ast.parse(source)
            
            # Find function calls
            calls = []
            
            class FunctionCallVisitor(ast.NodeVisitor):
                def visit_Call(self, node):
                    # Check if this is a call to our target function
                    if isinstance(node.func, ast.Name) and node.func.id == function_name:
                        calls.append({
                            'line': node.lineno,
                            'col': node.col_offset,
                            'args': len(node.args),
                            'keywords': {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
                        })
                    self.generic_visit(node)
            
            FunctionCallVisitor().visit(tree)
            return calls
            
        except Exception as e:
            self.logger.logger.error(f"Error analyzing function calls in {file_path}: {str(e)}")
            return []

# Example usage
if __name__ == "__main__":
    integrator = DebugIntegrator()
    
    # Analyze a file
    file_path = "trading_bot/main.py"  # Replace with an actual file path
    analysis = integrator.analyze_file(file_path)
    
    # Generate and apply logging code
    if analysis:
        modified_code = integrator.apply_logging(file_path, dry_run=True)
        print(f"Generated logging code for {len(analysis.get('functions', []))} functions and {len(analysis.get('classes', []))} classes.") 