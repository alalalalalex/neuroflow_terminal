"""
NeuroFlow Command History Manager
Manages command history for the NeuroFlow terminal.
"""

from prompt_toolkit.history import FileHistory
import os


class HistoryManager:
    """Manages command history for the NeuroFlow terminal."""
    
    def __init__(self, history_file='.neuroflow_history'):
        self.history_file = history_file
        # Create a file-based history manager
        self.history = FileHistory(history_file)
    
    def get_history(self):
        """Return the history object."""
        return self.history
    
    def add_entry(self, command: str):
        """Add a command to the history."""
        # FileHistory automatically handles adding entries when used with PromptSession
        pass
    
    def clear_history(self):
        """Clear the command history."""
        if os.path.exists(self.history_file):
            os.remove(self.history_file)
            # Recreate the history object
            self.history = FileHistory(self.history_file)
    
    def load_history(self):
        """Load command history from file."""
        # FileHistory loads automatically when instantiated
        pass