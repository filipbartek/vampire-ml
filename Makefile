grammar = TPTP-ANTLR4-Grammar/tptp_v7_0_0_0.g4

tptp_v7_0_0_0: $(grammar)
	antlr4 -o tptp_v7_0_0_0 -Xexact-output-dir -visitor -no-listener -Dlanguage=Python3 $(grammar)
