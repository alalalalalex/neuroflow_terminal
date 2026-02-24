"""
NeuroFlow Lexer
Provides syntax highlighting for the NeuroFlow domain-specific language.
"""

from pygments.lexer import RegexLexer, words
from pygments.token import Keyword, Name, Number, String, Operator, Punctuation, Text, Comment

class NeuroFlowLexer(RegexLexer):
    name = 'NeuroFlow'
    aliases = ['neuroflow']
    filenames = ['*.nf']

    keywords = [
        'input', 'model', 'def', 'return', 'if', 'else', 'for', 'while',
        'int', 'float', 'bool', 'tensor', 'as'
    ]

    functions = [
        'dense', 'conv', 'pool', 'dropout', 'batch_norm', 'relu', 'softmax',
        'sigmoid', 'tanh', 'flatten', 'concat', 'add', 'multiply'
    ]

    tokens = {
        'root': [
            (r'\s+', Text.Whitespace),
            (r'#.*?$', Comment.Single),
            
            (words(keywords, suffix=r'\b'), Keyword),
            (words(functions, suffix=r'\b'), Name.Function),
            
            (r'[a-zA-Z_][a-zA-Z0-9_]*', Name.Variable),
            (r'\d+\.\d+', Number.Float),
            (r'\d+', Number.Integer),
            
            (r'"[^"]*"', String.Double),
            (r"'[^']*'", String.Single),
            
            (r'[+\-*/%=<>!]+', Operator),
            (r'[{}[\]();,.]', Punctuation),
            
            (r'.', Text),
        ],
    }