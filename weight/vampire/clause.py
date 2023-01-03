import logging
from collections import Counter

from antlr4 import *
from tptp_grammar.cnf_formulaLexer import cnf_formulaLexer as Lexer
from tptp_grammar.cnf_formulaParser import cnf_formulaParser as Parser
from tptp_grammar.cnf_formulaListener import cnf_formulaListener as Listener

from questions.utils import timer

log = logging.getLogger(__name__)


def token_counts(c, max_terminals=None):
    input_stream = InputStream(c)
    lexer = Lexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = Parser(stream)
    parser.buildParseTrees = False
    listener = MyListener(max_terminals=max_terminals)
    parser.addParseListener(listener)
    with timer() as t:
        parser.cnf_formula()
    if parser.getNumberOfSyntaxErrors() > 0:
        raise ValueError(f'{parser.getNumberOfSyntaxErrors()} syntax errors occurred while parsing \"{c}\".')
    assert stream.index == stream.getNumberOfOnChannelTokens() - 1 == listener.all_terminals
    res = {
        'literal': listener.literals,
        'not': listener.terminals[Parser.Not],
        'equality': listener.terminals[Parser.Infix_equality],
        'inequality': listener.terminals[Parser.Infix_inequality],
        'variable': sorted(listener.variables.values(), reverse=True),
        'number': listener.numbers,
        'symbol': listener.functors,
        'terminals': stream.index,
        'time': t.elapsed
    }
    return res


class MyListener(Listener):
    def __init__(self, max_terminals=None):
        super().__init__()
        self.max_terminals = max_terminals
        self.terminals = Counter()
        self.functors = Counter()
        self.variables = Counter()
        self.literals = 0
        self.numbers = 0
        self.all_terminals = 0

    tracked_terminal_types = [
        Parser.Not,
        Parser.Infix_equality,
        Parser.Infix_inequality,
    ]

    def visitTerminal(self, node:TerminalNode):
        self.all_terminals += 1
        if self.max_terminals is not None and self.all_terminals > self.max_terminals:
            raise MaxTerminalsError(self.max_terminals)
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


class MaxTerminalsError(ValueError):
    def __init__(self, max_terminals):
        super().__init__(f'The formula has more than {max_terminals} terminals.')
