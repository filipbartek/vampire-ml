from collections import Counter

from antlr4 import *
from tptp_grammar.tptp_v7_0_0_0Lexer import tptp_v7_0_0_0Lexer
from tptp_grammar.tptp_v7_0_0_0Parser import tptp_v7_0_0_0Parser
from tptp_grammar.tptp_v7_0_0_0Listener import tptp_v7_0_0_0Listener


def token_counts(c):
    input_stream = InputStream(c)
    lexer = tptp_v7_0_0_0Lexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = tptp_v7_0_0_0Parser(stream)
    listener = Listener()
    tree = parser.cnf_formula()
    if parser.getNumberOfSyntaxErrors() > 0:
        raise RuntimeError(f'{parser.getNumberOfSyntaxErrors()} syntax errors occurred while parsing \"{c}\".')
    walker = ParseTreeWalker()
    walker.walk(listener, tree)

    res = {
        'literal': listener.literals,
        'not': listener.terminals[tptp_v7_0_0_0Parser.Not],
        'equality': listener.terminals[tptp_v7_0_0_0Parser.Infix_equality],
        'inequality': listener.terminals[tptp_v7_0_0_0Parser.Infix_inequality],
        'variable': sorted(listener.variables.values(), reverse=True),
        'number': listener.numbers,
        'symbol': listener.functors,
    }
    return res


class Listener(tptp_v7_0_0_0Listener):
    def __init__(self):
        super().__init__()
        self.terminals = Counter()
        self.functors = Counter()
        self.variables = Counter()
        self.literals = 0
        self.numbers = 0

    tracked_terminal_types = [
        tptp_v7_0_0_0Parser.Not,
        tptp_v7_0_0_0Parser.Infix_equality,
        tptp_v7_0_0_0Parser.Infix_inequality,
    ]

    def visitTerminal(self, node:TerminalNode):
        symbol_type = node.symbol.type
        if symbol_type in self.tracked_terminal_types:
            self.terminals[symbol_type] += 1

    def enterFunctor(self, ctx):
        self.functors[ctx.start.text] += 1

    def enterVariable(self, ctx):
        self.variables[ctx.start.text] += 1

    def enterCnf_literal(self, ctx):
        self.literals += 1

    def enterNumber(self, ctx):
        self.numbers += 1
