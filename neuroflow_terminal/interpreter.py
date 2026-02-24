"""
NeuroFlow Interpreter
Executes the parsed AST and manages the neural network model state.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional


class NeuroFlowInterpreter:
    """Interprets and executes the parsed AST to build and manage neural network models."""
    
    def __init__(self):
        self.inputs = {}  # Store input definitions
        self.models = {}  # Store defined models
        self.current_model = None  # Currently active model
        self.variables = {}  # Store variable values during execution
        
    def execute(self, ast_node) -> str:
        """Execute an AST node and return a result message."""
        node_type = ast_node.type
        
        if node_type == 'input':
            return self.execute_input(ast_node)
        elif node_type == 'model_def':
            return self.execute_model_definition(ast_node)
        elif node_type == 'assignment':
            return self.execute_assignment(ast_node)
        elif node_type == 'expression':
            return self.execute_expression(ast_node)
        elif node_type == 'statement':
            # For generic statements, just return success
            return "Code executed successfully"
        else:
            raise ValueError(f"Unknown AST node type: {node_type}")
    
    def execute_input(self, node) -> str:
        """Execute an input definition."""
        name = node.name
        shape = node.shape
        
        self.inputs[name] = {
            'name': name,
            'shape': shape,
            'type': 'input'
        }
        
        return f"Input '{name}' with shape {shape} defined successfully"
    
    def execute_model_definition(self, node) -> str:
        """Execute a model definition."""
        model_name = node.name
        layers = node.layers
        
        # Create a new model
        model = nn.Module()
        layer_modules = nn.ModuleList()
        layer_info = []
        
        # Process each layer in the model definition
        for i, layer_node in enumerate(layers):
            layer_name = layer_node.name
            operation = layer_node.operation
            activation = layer_node.activation
            
            # Create the layer based on the operation
            layer_instance = self.create_layer(operation, layer_name)
            layer_modules.append(layer_instance)
            
            # Store layer information
            layer_info.append({
                'name': layer_name,
                'operation': operation.name,
                'activation': activation,
                'index': i
            })
        
        # Add layers to the model
        for i, layer in enumerate(layer_modules):
            setattr(model, f"layer_{i}", layer)
        
        # Store the model
        self.models[model_name] = {
            'model': model,
            'layers': layer_info,
            'inputs': self.inputs.copy()
        }
        
        self.current_model = self.models[model_name]
        
        return f"Model '{model_name}' defined with {len(layers)} layers"
    
    def create_layer(self, operation_node, layer_name):
        """Create a PyTorch layer based on the operation."""
        op_name = operation_node.name
        args = operation_node.args
        
        if op_name == 'dense':
            # Dense layer: dense(input_size, output_size)
            if len(args) >= 2:
                input_size = int(args[0].value) if hasattr(args[0], 'value') else 1
                output_size = int(args[1].value) if hasattr(args[1], 'value') else 1
                return nn.Linear(input_size, output_size)
            else:
                raise ValueError(f"Dense layer requires 2 arguments, got {len(args)}")
        elif op_name == 'conv':
            # Conv layer: conv(in_channels, out_channels, kernel_size)
            if len(args) >= 3:
                in_channels = int(args[0].value)
                out_channels = int(args[1].value)
                kernel_size = int(args[2].value)
                return nn.Conv2d(in_channels, out_channels, kernel_size)
            else:
                raise ValueError(f"Conv layer requires 3 arguments, got {len(args)}")
        elif op_name == 'flatten':
            # Flatten layer: no arguments needed
            return nn.Flatten()
        elif op_name == 'relu':
            return nn.ReLU()
        elif op_name == 'softmax':
            return nn.Softmax(dim=-1)
        elif op_name == 'sigmoid':
            return nn.Sigmoid()
        elif op_name == 'tanh':
            return nn.Tanh()
        elif op_name == 'dropout':
            # Dropout layer: dropout(probability)
            if len(args) >= 1:
                prob = float(args[0].value) if hasattr(args[0], 'value') else 0.5
                return nn.Dropout(prob)
            else:
                return nn.Dropout()
        elif op_name == 'batch_norm':
            # Batch norm layer: batch_norm(num_features)
            if len(args) >= 1:
                num_features = int(args[0].value) if hasattr(args[0], 'value') else 1
                return nn.BatchNorm1d(num_features)
            else:
                raise ValueError(f"Batch norm requires 1 argument, got {len(args)}")
        elif op_name == 'pool':
            # Max pooling layer: pool(kernel_size)
            if len(args) >= 1:
                kernel_size = int(args[0].value) if hasattr(args[0], 'value') else 2
                return nn.MaxPool2d(kernel_size)
            else:
                return nn.MaxPool2d(2)
        else:
            # For unknown operations, return a placeholder
            class PlaceholderLayer(nn.Module):
                def __init__(self, name, args):
                    super().__init__()
                    self.name = name
                    self.args = args
                
                def forward(self, x):
                    return x  # Identity function for unknown layers
            
            return PlaceholderLayer(op_name, args)
    
    def execute_assignment(self, node) -> str:
        """Execute an assignment statement."""
        variable = node.variable
        value = node.value
        
        # For now, we'll store the AST node as the value
        self.variables[variable] = value
        
        return f"Variable '{variable}' assigned"
    
    def execute_expression(self, node) -> str:
        """Execute an expression."""
        # For now, just return the value
        if hasattr(node, 'value'):
            return f"Expression evaluated to: {node.value}"
        else:
            return "Expression evaluated"
    
    def get_current_model(self):
        """Return the currently active model."""
        return self.current_model
    
    def print_layer_stats(self):
        """Print statistics about the layers in the current model."""
        if not self.current_model:
            print("No model defined yet.")
            return
        
        model = self.current_model['model']
        layers_info = self.current_model['layers']
        
        print("📊 Layer Statistics:")
        print("=" * 60)
        
        total_params = 0
        for info in layers_info:
            layer_idx = info['index']
            layer_name = info['name']
            layer_op = info['operation']
            
            # Get the actual layer module
            layer_module = getattr(model, f"layer_{layer_idx}")
            
            # Count parameters
            param_count = sum(p.numel() for p in layer_module.parameters() if p.requires_grad)
            total_params += param_count
            
            # Try to get input/output shapes (this is approximate)
            # For demonstration purposes, we'll show placeholders
            print(f"  {layer_name:<20} | Params: {param_count:>10} | Operation: {layer_op}")
        
        print("=" * 60)
        print(f"Total Parameters: {total_params:,}")
    
    def save_session(self, filename: str):
        """Save the current session to a file."""
        session_data = {
            'inputs': self.inputs,
            'models': {},
            'variables': self.variables
        }
        
        # Serialize models separately since nn.Module can't be directly serialized
        for model_name, model_data in self.models.items():
            session_data['models'][model_name] = {
                'state_dict': {k: v.tolist() if isinstance(v, torch.Tensor) else v 
                              for k, v in model_data['model'].state_dict().items()},
                'layers': model_data['layers'],
                'inputs': model_data['inputs']
            }
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"✅ Session saved to {filename}")
    
    def load_session(self, filename: str):
        """Load a session from a file."""
        if not os.path.exists(filename):
            print(f"❌ File {filename} does not exist")
            return
        
        with open(filename, 'r') as f:
            session_data = json.load(f)
        
        # Restore inputs
        self.inputs = session_data.get('inputs', {})
        
        # Restore models
        self.models = {}
        for model_name, model_data in session_data.get('models', {}).items():
            # Recreate the model structure
            model = nn.Module()
            layer_modules = nn.ModuleList()
            
            # Rebuild layers based on stored info
            for i, layer_info in enumerate(model_data['layers']):
                layer_instance = self.create_layer_from_info(layer_info)
                layer_modules.append(layer_instance)
            
            # Add layers to the model
            for i, layer in enumerate(layer_modules):
                setattr(model, f"layer_{i}", layer)
            
            # Load state dict if available
            state_dict = model_data.get('state_dict', {})
            # Convert lists back to tensors
            state_dict_tensors = {}
            for k, v in state_dict.items():
                if isinstance(v, list):
                    state_dict_tensors[k] = torch.tensor(v)
                else:
                    state_dict_tensors[k] = v
            
            model.load_state_dict(state_dict_tensors, strict=False)
            
            self.models[model_name] = {
                'model': model,
                'layers': model_data['layers'],
                'inputs': model_data['inputs']
            }
        
        # Restore variables
        self.variables = session_data.get('variables', {})
        
        # Set current model to the first one if available
        if self.models:
            first_model_name = next(iter(self.models))
            self.current_model = self.models[first_model_name]
        
        print(f"✅ Session loaded from {filename}")
    
    def create_layer_from_info(self, layer_info):
        """Helper method to recreate a layer from stored information."""
        op_name = layer_info['operation']
        
        # For this basic implementation, we'll create a placeholder
        # A full implementation would need to store and restore layer parameters
        if op_name == 'Linear':
            return nn.Linear(1, 1)  # Placeholder
        elif op_name == 'Conv2d':
            return nn.Conv2d(1, 1, 1)  # Placeholder
        elif op_name == 'Flatten':
            return nn.Flatten()
        elif op_name == 'ReLU':
            return nn.ReLU()
        elif op_name == 'Softmax':
            return nn.Softmax(dim=-1)
        elif op_name == 'Sigmoid':
            return nn.Sigmoid()
        elif op_name == 'Tanh':
            return nn.Tanh()
        elif op_name == 'Dropout':
            return nn.Dropout()
        elif op_name == 'BatchNorm1d':
            return nn.BatchNorm1d(1)  # Placeholder
        elif op_name == 'MaxPool2d':
            return nn.MaxPool2d(2)  # Placeholder
        else:
            class PlaceholderLayer(nn.Module):
                def __init__(self, name):
                    super().__init__()
                    self.name = name
                
                def forward(self, x):
                    return x  # Identity function for unknown layers
            
            return PlaceholderLayer(op_name)