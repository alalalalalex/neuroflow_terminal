"""
NeuroFlow Parser
Parses the NeuroFlow domain-specific language into an Abstract Syntax Tree (AST).
"""

import re
from typing import Dict, List, Any, Union


class ASTNode:
    """Base class for all AST nodes."""
    def __init__(self, node_type: str, **kwargs):
        self.type = node_type
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        attrs = []
        for key, value in self.__dict__.items():
            if key != 'type':
                attrs.append(f"{key}={repr(value)}")
        return f"ASTNode({self.type}, {', '.join(attrs)})"


class NeuroFlowParser:
    """Parses NeuroFlow DSL code into an AST."""
    
    def __init__(self):
        self.tokens = []
        self.pos = 0
        
    def tokenize(self, code: str) -> List[Dict[str, Any]]:
        """Convert source code into a list of tokens."""
        # Define token patterns
        token_patterns = [
            ('INPUT', r'input\b'),
            ('MODEL', r'model\b'),
            ('DEF', r'def\b'),
            ('RETURN', r'return\b'),
            ('IF', r'if\b'),
            ('ELSE', r'else\b'),
            ('FOR', r'for\b'),
            ('WHILE', r'while\b'),
            ('INT_TYPE', r'int\b'),
            ('FLOAT_TYPE', r'float\b'),
            ('BOOL_TYPE', r'bool\b'),
            ('TENSOR_TYPE', r'tensor\b'),
            ('AS', r'as\b'),
            ('IDENTIFIER', r'[a-zA-Z_][a-zA-Z0-9_]*'),
            ('NUMBER', r'\d+(\.\d+)?'),
            ('STRING', r'"[^"]*"|\'[^\']*\''),
            ('LBRACE', r'{'),
            ('RBRACE', r'}'),
            ('LPAREN', r'\('),
            ('RPAREN', r'\)'),
            ('LBRACKET', r'\['),
            ('RBRACKET', r'\]'),
            ('COMMA', r','),
            ('SEMICOLON', r';'),
            ('ASSIGN', r'='),
            ('ARROW', r'->'),
            ('PLUS', r'\+'),
            ('MINUS', r'-'),
            ('MULT', r'\*'),
            ('DIV', r'/'),
            ('EQUAL', r'=='),
            ('NOT_EQUAL', r'!='),
            ('LESS_THAN', r'<'),
            ('GREATER_THAN', r'>'),
            ('LESS_EQUAL', r'<='),
            ('GREATER_EQUAL', r'>='),
            ('COMMENT', r'#.*'),
            ('WHITESPACE', r'\s+'),
        ]
        
        # Compile the regex pattern
        combined_pattern = '|'.join(f"(?P<{name}>{pattern})" for name, pattern in token_patterns)
        token_regex = re.compile(combined_pattern)
        
        tokens = []
        pos = 0
        
        while pos < len(code):
            match = token_regex.match(code, pos)
            if match:
                token_type = match.lastgroup
                token_value = match.group(token_type)
                
                # Skip whitespace and comments
                if token_type not in ['WHITESPACE', 'COMMENT']:
                    tokens.append({'type': token_type, 'value': token_value})
                
                pos = match.end()
            else:
                # Handle unrecognized characters
                raise SyntaxError(f"Unexpected character at position {pos}: '{code[pos]}'")
        
        return tokens
    
    def parse(self, code: str) -> ASTNode:
        """Parse the code and return the AST root node."""
        self.tokens = self.tokenize(code)
        self.pos = 0
        
        # Determine the type of statement based on the first token
        if self.current_token() and self.current_token()['type'] == 'INPUT':
            return self.parse_input_statement()
        elif self.current_token() and self.current_token()['type'] == 'MODEL':
            return self.parse_model_definition()
        elif self.current_token() and self.current_token()['type'] == 'IDENTIFIER':
            # Could be an assignment or function call
            return self.parse_assignment_or_expression()
        else:
            # For now, just return a generic statement node
            return ASTNode('statement', code=code)
    
    def current_token(self):
        """Get the current token without advancing the position."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None
    
    def next_token(self):
        """Advance to the next token and return it."""
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            self.pos += 1
            return token
        return None
    
    def expect_token(self, expected_type: str):
        """Expect a specific token type and advance to the next token."""
        token = self.next_token()
        if not token or token['type'] != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {token['type'] if token else 'EOF'}")
        return token
    
    def parse_input_statement(self) -> ASTNode:
        """Parse an input statement like 'input image [28, 28]'."""
        self.expect_token('INPUT')  # consume 'input'
        
        identifier_token = self.expect_token('IDENTIFIER')
        name = identifier_token['value']
        
        shape = self.parse_shape()
        
        return ASTNode('input', name=name, shape=shape)
    
    def parse_shape(self) -> List[int]:
        """Parse a shape like [28, 28]."""
        self.expect_token('LBRACKET')
        
        shape = []
        while True:
            num_token = self.expect_token('NUMBER')
            shape.append(int(num_token['value']))
            
            if self.current_token() and self.current_token()['type'] == 'COMMA':
                self.next_token()  # consume comma
            else:
                break
        
        self.expect_token('RBRACKET')
        return shape
    
    def parse_model_definition(self) -> ASTNode:
        """Parse a model definition like 'model MyNet { ... }'."""
        self.expect_token('MODEL')  # consume 'model'
        
        name_token = self.expect_token('IDENTIFIER')
        model_name = name_token['value']
        
        self.expect_token('LBRACE')  # consume '{'
        
        layers = []
        while self.current_token() and self.current_token()['type'] != 'RBRACE':
            layer = self.parse_layer()
            layers.append(layer)
            
            # Skip semicolons if present
            if self.current_token() and self.current_token()['type'] == 'SEMICOLON':
                self.next_token()
        
        self.expect_token('RBRACE')  # consume '}'
        
        return ASTNode('model_def', name=model_name, layers=layers)
    
    def parse_layer(self) -> ASTNode:
        """Parse a layer definition like 'h1 = dense(x, 128) -> relu'."""
        # Parse left-hand side (variable name)
        var_token = self.expect_token('IDENTIFIER')
        var_name = var_token['value']
        
        self.expect_token('ASSIGN')  # consume '='
        
        # Parse function call like 'dense(x, 128)'
        func_call = self.parse_function_call()
        
        activation = None
        # Check if there's an arrow operator for activation
        if self.current_token() and self.current_token()['type'] == 'ARROW':
            self.next_token()  # consume '->'
            activation_token = self.expect_token('IDENTIFIER')
            activation = activation_token['value']
        
        return ASTNode('layer', name=var_name, operation=func_call, activation=activation)
    
    def parse_function_call(self) -> ASTNode:
        """Parse a function call like 'dense(x, 128)'."""
        func_token = self.expect_token('IDENTIFIER')
        func_name = func_token['value']
        
        self.expect_token('LPAREN')  # consume '('
        
        args = []
        while self.current_token() and self.current_token()['type'] != 'RPAREN':
            arg = self.parse_expression()
            args.append(arg)
            
            if self.current_token() and self.current_token()['type'] == 'COMMA':
                self.next_token()  # consume comma
            else:
                break
        
        self.expect_token('RPAREN')  # consume ')'
        
        return ASTNode('function_call', name=func_name, args=args)
    
    def parse_expression(self):
        """Parse a simple expression."""
        # For now, just handle identifiers and numbers
        token = self.next_token()
        if token['type'] in ['IDENTIFIER', 'NUMBER']:
            return ASTNode('literal', value=token['value'])
        else:
            raise SyntaxError(f"Unexpected token in expression: {token['type']}")
    
    def parse_assignment_or_expression(self) -> ASTNode:
        """Parse an assignment or expression starting with an identifier."""
        first_token = self.current_token()
        peek_next = self.lookahead(1)
        
        if peek_next and peek_next['type'] == 'ASSIGN':
            # It's an assignment
            var_token = self.next_token()
            self.expect_token('ASSIGN')
            
            expr = self.parse_expression()
            
            return ASTNode('assignment', variable=var_token['value'], value=expr)
        else:
            # Just an expression
            expr = self.parse_expression()
            return ASTNode('expression', value=expr)
    
    def lookahead(self, n: int = 1):
        """Look ahead at the nth token without consuming it."""
        if self.pos + n < len(self.tokens):
            return self.tokens[self.pos + n]
        return None