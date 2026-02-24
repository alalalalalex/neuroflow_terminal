"""
NeuroFlow Auto-Completer
Provides auto-completion functionality for the NeuroFlow DSL.
"""

from prompt_toolkit.completion import Completer, Completion
from typing import List


class NeuroFlowCompleter(Completer):
    """Completer for NeuroFlow DSL keywords and functions."""
    
    def __init__(self):
        # Define possible completions
        self.keywords = [
            'input', 'model', 'def', 'return', 'if', 'else', 'for', 'while',
            'int', 'float', 'bool', 'tensor', 'as'
        ]
        
        self.functions = [
            'dense', 'conv', 'pool', 'dropout', 'batch_norm', 'relu', 'softmax',
            'sigmoid', 'tanh', 'flatten', 'concat', 'add', 'multiply'
        ]
        
        self.special_commands = [
            '/help', '/model', '/layers', '/save', '/load', '/clear', '/exit', '/quit'
        ]
    
    def get_completions(self, document, complete_event):
        """Generate completions based on the current input."""
        # Get the current word being typed
        word_before_cursor = document.get_word_before_cursor(WORD=True)
        
        # Check if we're completing a special command
        if word_before_cursor.startswith('/'):
            for cmd in self.special_commands:
                if cmd.startswith(word_before_cursor):
                    yield Completion(cmd, start_position=-len(word_before_cursor))
        else:
            # Complete keywords
            for keyword in self.keywords:
                if keyword.startswith(word_before_cursor):
                    yield Completion(keyword, start_position=-len(word_before_cursor))
            
            # Complete functions
            for func in self.functions:
                if func.startswith(word_before_cursor):
                    yield Completion(func, start_position=-len(word_before_cursor))