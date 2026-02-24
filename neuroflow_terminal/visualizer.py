"""
NeuroFlow Model Visualizer
Provides ASCII visualization of neural network architectures.
"""

from typing import Dict, Any, Optional


class ModelVisualizer:
    """Visualizes neural network models in ASCII format."""
    
    def __init__(self):
        pass
    
    def display_model(self, model_data: Optional[Dict[str, Any]]) -> None:
        """Display the model architecture in ASCII format."""
        if not model_data:
            print("No model to visualize.")
            return
        
        model_name = model_data.get('name', 'Unknown')
        layers = model_data.get('layers', [])
        inputs = model_data.get('inputs', {})
        
        # Calculate total parameters
        total_params = self._calculate_total_params(model_data)
        
        # Print header
        print()
        self._print_header(f"Model: {model_name}")
        
        # Print inputs
        print("║  INPUTS:                                                     ║")
        if inputs:
            for input_name, input_info in inputs.items():
                shape_str = str(input_info.get('shape', 'unknown'))
                print(f"║    📥 {input_name:<18} {shape_str:<30} ║")
        else:
            print("║    No inputs defined                                        ║")
        
        # Print layers
        print("║  LAYERS:                                                     ║")
        if layers:
            for i, layer in enumerate(layers, 1):
                layer_name = layer.get('name', f'layer_{i}')
                operation = layer.get('operation', 'unknown')
                
                # Map operations to icons
                icon = self._get_layer_icon(operation)
                
                print(f"║    {i}. {icon} {layer_name:<15} [{operation:<15}]           ║")
        else:
            print("║    No layers defined                                        ║")
        
        # Print total parameters
        params_str = f"{total_params:,}".replace(',', ' ')
        print(f"║  TOTAL PARAMETERS: {params_str:<37} ║")
        
        # Print footer
        self._print_footer()
        print()
    
    def _calculate_total_params(self, model_data: Dict[str, Any]) -> int:
        """Calculate the total number of parameters in the model."""
        # In a real implementation, this would calculate from the actual model
        # For this example, we'll return a placeholder value
        # A full implementation would need to extract parameter counts from the PyTorch model
        return sum(range(len(model_data.get('layers', [])))) * 1000  # Placeholder
    
    def _get_layer_icon(self, operation: str) -> str:
        """Get an appropriate icon for the layer operation."""
        icon_map = {
            'dense': '🔵',
            'linear': '🔵',
            'conv': '🔳',
            'conv2d': '🔳',
            'flatten': '📏',
            'relu': '⚡',
            'softmax': '🔥',
            'sigmoid': '🌀',
            'tanh': '🌊',
            'dropout': '💧',
            'batch_norm': '⚖️',
            'pool': '🔽',
            'max_pool': '🔽',
            'avg_pool': '🔽'
        }
        
        # Return the icon if found, otherwise return a generic one
        return icon_map.get(operation.lower(), '🔷')
    
    def _print_header(self, title: str) -> None:
        """Print the top border of the visualization box."""
        print("╔══════════════════════════════════════════════════════════════╗")
        title_formatted = f"║  {title:<57} ║"
        print(title_formatted)
        print("╚══════════════════════════════════════════════════════════════╝")
        print("╔══════════════════════════════════════════════════════════════╗")
    
    def _print_footer(self) -> None:
        """Print the bottom border of the visualization box."""
        print("╚══════════════════════════════════════════════════════════════╝")