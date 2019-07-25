#!/usr/bin/env python3.7

import argparse
import json

import antlr4

from tptp_v7_0_0_0.tptp_v7_0_0_0Lexer import tptp_v7_0_0_0Lexer
from tptp_v7_0_0_0.tptp_v7_0_0_0Parser import tptp_v7_0_0_0Parser
from tptp_v7_0_0_0.tptp_v7_0_0_0Visitor import tptp_v7_0_0_0Visitor


# TODO: Finish the implementation.
class CnfVisitor(tptp_v7_0_0_0Visitor):
    def aggregateResult(self, aggregate, nextResult):
        if aggregate is None:
            return [nextResult]
        return aggregate + [nextResult]

    def visitTerminal(self, node):
        return {
            'text': node.symbol.text,
            'type': node.symbol.type
        }

    def visitVariable(self, ctx: tptp_v7_0_0_0Parser.VariableContext):
        res = self.visitChildren(ctx)
        assert len(res) == 1
        assert res[0]['type'] == tptp_v7_0_0_0Lexer.Upper_word
        return {
            'type': 'variable',
            'name': res[0]['text']
        }

    def visitFof_term(self, ctx: tptp_v7_0_0_0Parser.Fof_termContext):
        res = self.visitChildren(ctx)
        assert len(res) == 1
        return res[0]

    def visitFunctor(self, ctx: tptp_v7_0_0_0Parser.FunctorContext):
        res = self.visitChildren(ctx)
        assert len(res) == 1
        return {
            'type': 'functor',
            'name': res[0]['text']
        }

    def visitAtomic_word(self, ctx: tptp_v7_0_0_0Parser.Atomic_wordContext):
        res = self.visitChildren(ctx)
        assert len(res) == 1
        assert res[0]['type'] in [tptp_v7_0_0_0Lexer.Lower_word, tptp_v7_0_0_0Lexer.Single_quoted]
        return res[0]

    def visitFof_plain_term(self, ctx: tptp_v7_0_0_0Parser.Fof_plain_termContext):
        res = self.visitChildren(ctx)
        assert len(res) == 1 or len(res) == 4
        if len(res) == 1:
            return {
                'type': 'fof_plain_term',
                'functor': res[0],
                'arguments': []
            }
        return {
            'type': 'fof_plain_term',
            'functor': res[0],
            'arguments': res[2]
        }

    def visitCnf_disjunction(self, ctx: tptp_v7_0_0_0Parser.Cnf_disjunctionContext):
        res = self.visitChildren(ctx)
        return {
            'value': res
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=argparse.FileType('r'), help='list of result files of problem runs')
    parser.add_argument('--tptp_syntax_bnf', type=argparse.FileType('r'), required=True, help='TPTP syntax BNF')
    namespace = parser.parse_args()

    visitor = CnfVisitor()

    for line in namespace.input:
        path = line.strip()
        if path:
            with open(path, 'r') as data_file:
                data = json.load(data_file)
                problem_path = data['parameters']['paths']['problem']
                print(problem_path)
                input_stream = antlr4.FileStream(problem_path)
                lexer = tptp_v7_0_0_0Lexer(input_stream)
                stream = antlr4.CommonTokenStream(lexer)
                parser = tptp_v7_0_0_0Parser(stream)
                tree = parser.tptp_file()
                problem_internal = visitor.visit(tree)
                print(problem_internal)

    namespace.input.close()
