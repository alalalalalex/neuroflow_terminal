#!/usr/bin/env python3
"""
NeuroFlow Interactive Terminal (REPL)
Interactive console with syntax highlighting, auto-completion, history, and real-time model visualization.
"""

import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from pygments.lexers import PythonLexer
from lexer import NeuroFlowLexer
from parser import NeuroFlowParser
from interpreter import NeuroFlowInterpreter
from visualizer import ModelVisualizer
from completer import NeuroFlowCompleter
from history import HistoryManager


class NeuroFlowTerminal:
    def __init__(self):
        self.parser = NeuroFlowParser()
        self.interpreter = NeuroFlowInterpreter()
        self.visualizer = ModelVisualizer()
        self.history_manager = HistoryManager()
        self.completer = NeuroFlowCompleter()
        
        # Initialize the prompt session
        self.session = PromptSession(
            history=self.history_manager.get_history(),
            completer=self.completer,
            lexer=PygmentsLexer(NeuroFlowLexer),
            style=Style.from_dict({
                'prompt': '#ansiblue',
                'completion-menu': 'bg:#ansiyellow #ansiwhite',
                'completion-menu.completion.current': 'bg:#ansiblue #ansiwhite',
            })
        )
        
        self.running = True
        
    def print_banner(self):
        """Display the welcome banner."""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║           🧠 NeuroFlow Interactive Terminal v1.0             ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def handle_special_command(self, cmd):
        """Handle special commands starting with '/'."""
        if cmd == '/help':
            self.show_help()
        elif cmd == '/exit' or cmd == '/quit':
            self.exit_terminal()
        elif cmd.startswith('/save'):
            parts = cmd.split(' ', 1)
            if len(parts) == 2:
                self.interpreter.save_session(parts[1])
            else:
                print("Usage: /save <filename>")
        elif cmd.startswith('/load'):
            parts = cmd.split(' ', 1)
            if len(parts) == 2:
                self.interpreter.load_session(parts[1])
            else:
                print("Usage: /load <filename>")
        elif cmd == '/model':
            self.visualizer.display_model(self.interpreter.get_current_model())
        elif cmd == '/layers':
            self.interpreter.print_layer_stats()
        elif cmd == '/clear':
            import os
            os.system('clear' if os.name != 'nt' else 'cls')
        else:
            print(f"Unknown command: {cmd}. Type /help for available commands.")
    
    def show_help(self):
        """Display help information."""
        help_text = """
Available commands:
  /help      - Show this help message
  /model     - Display current model architecture
  /layers    - Show layer statistics
  /save <fn> - Save current session to file
  /load <fn> - Load session from file
  /clear     - Clear the terminal screen
  /exit      - Exit the terminal
        """
        print(help_text)
    
    def exit_terminal(self):
        """Exit the terminal."""
        print("\n👋 Goodbye!")
        self.running = False
    
    def run(self):
        """Main loop of the terminal."""
        self.print_banner()
        
        while self.running:
            try:
                # Get input from user
                user_input = self.session.prompt('neuroflow> ', multiline=False)
                
                # Skip empty input
                if not user_input.strip():
                    continue
                
                # Check if it's a special command
                if user_input.strip().startswith('/'):
                    self.handle_special_command(user_input.strip())
                    continue
                
                # Parse and execute the input
                try:
                    ast = self.parser.parse(user_input)
                    result = self.interpreter.execute(ast)
                    
                    if result:
                        print(f"✅ {result}")
                    
                    # Visualize the model if it was updated
                    current_model = self.interpreter.get_current_model()
                    if current_model:
                        self.visualizer.display_model(current_model)
                        
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
                    
            except KeyboardInterrupt:
                print("\nUse /exit to quit the terminal.")
            except EOFError:
                self.exit_terminal()


if __name__ == "__main__":
    terminal = NeuroFlowTerminal()
    terminal.run()