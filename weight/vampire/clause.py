from collections import Counter

from antlr4 import *
from tptp_grammar.tptp_v7_0_0_0Lexer import tptp_v7_0_0_0Lexer as Lexer
from tptp_grammar.tptp_v7_0_0_0Parser import tptp_v7_0_0_0Parser as Parser
from tptp_grammar.tptp_v7_0_0_0Listener import tptp_v7_0_0_0Listener as Listener


def token_counts(c):
    input_stream = InputStream(c)
    lexer = Lexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = Parser(stream)
    parser.buildParseTrees = False
    listener = MyListener()
    parser.addParseListener(listener)
    parser.cnf_formula()
    if parser.getNumberOfSyntaxErrors() > 0:
        raise ValueError(f'{parser.getNumberOfSyntaxErrors()} syntax errors occurred while parsing \"{c}\".')

    res = {
        'literal': listener.literals,
        'not': listener.terminals[Parser.Not],
        'equality': listener.terminals[Parser.Infix_equality],
        'inequality': listener.terminals[Parser.Infix_inequality],
        'variable': sorted(listener.variables.values(), reverse=True),
        'number': listener.numbers,
        'symbol': listener.functors,
    }
    return res


class MyListener(Listener):
    def __init__(self):
        super().__init__()
        self.terminals = Counter()
        self.functors = Counter()
        self.variables = Counter()
        self.literals = 0
        self.numbers = 0

    tracked_terminal_types = [
        Parser.Not,
        Parser.Infix_equality,
        Parser.Infix_inequality,
    ]

    def visitTerminal(self, node:TerminalNode):
        symbol_type = node.symbol.type
        if symbol_type in self.tracked_terminal_types:
            self.terminals[symbol_type] += 1
        if isinstance(node.parentCtx, Parser.Atomic_wordContext) and isinstance(node.parentCtx.parentCtx, Parser.FunctorContext):
            if symbol_type == Parser.Lower_word:
                symbol_text = node.symbol.text
            elif symbol_type == Parser.Single_quoted:
                assert node.symbol.text[0] == '\'' and node.symbol.text[-1] == '\''
                symbol_text = node.symbol.text[1:-1]
            else:
                raise ValueError(f'Unsupported type of functor token: {symbol_type}')
            self.functors[symbol_text] += 1

    def enterVariable(self, ctx):
        self.variables[ctx.start.text] += 1

    def enterCnf_literal(self, ctx):
        self.literals += 1

    def enterNumber(self, ctx):
        self.numbers += 1
