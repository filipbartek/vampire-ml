/*
Root rule: `cnf_formula`

Only the content transitively used by `cnf_formula` has been preserved from the original TPTP grammar.
Additionally, some cases of `fof_term` have been removed.

To generate the Python sources of Lexer, Parser and Listener, run:

```
antlr-4.9-complete.jar cnf_formula.g4 -Dlanguage=Python3
```

Original TPTP grammar:
https://github.com/TobiasScholl/TPTP-ANTLR4-Grammar/blob/e2e510bf4ec489732971347d34742a4f40652b98/tptp_v7_0_0_0.g4

Line numbers pointing to the corresponding position in the original grammar are included as line comments.
*/

grammar cnf_formula;

// 13
fragment Do_char : [\u0020-\u0021\u0023-\u005B\u005D-\u007E] | '\\'["\\];
fragment Sq_char : [\u0020-\u0026\u0028-\u005B\u005D-\u007E] | '\\'['\\];
fragment Sign : [+-];
fragment Exponent : [Ee];
fragment Non_zero_numeric : [1-9];
fragment Numeric : [0-9];
fragment Lower_alpha : [a-z];
fragment Upper_alpha : [A-Z];
fragment Alpha_numeric : Lower_alpha | Upper_alpha | Numeric | '_';

// 23
Or: '|';

// 31
Not: '~';

// 34
Infix_inequality : '!=';
Infix_equality : '=';

// 47
Assignment: ':=';

// 54
Real : Signed_real | Unsigned_real;
Signed_real : Sign Unsigned_real;
Unsigned_real : Decimal_fraction|Decimal_exponent;
Rational: Signed_rational | Unsigned_rational;
Signed_rational: Sign Unsigned_rational;
Unsigned_rational: Decimal '/' Positive_decimal;
Integer : Signed_integer | Unsigned_integer;
Signed_integer: Sign Unsigned_integer;
Unsigned_integer: Decimal;
Decimal : '0' | Positive_decimal;
Positive_decimal : Non_zero_numeric Numeric*;
Decimal_exponent : (Decimal|Decimal_fraction) Exponent Exp_integer;
Decimal_fraction : Decimal Dot_decimal;
Dot_decimal : '.' Numeric Numeric*;
Exp_integer : Signed_exp_integer|Unsigned_exp_integer;
Signed_exp_integer : Sign Unsigned_exp_integer;
Unsigned_exp_integer : Numeric Numeric*;

// 72
Dollar_word : '$' Lower_word;
Dollar_dollar_word : '$$' Lower_word;
Upper_word : Upper_alpha Alpha_numeric*;
Lower_word : Lower_alpha Alpha_numeric*;

// 77
Single_quoted : '\'' Sq_char+ '\'';
Distinct_object : '"' Do_char+ '"';

// 80
WS : [ \r\t\n]+ -> skip ;
Line_comment : '%' ~[\r\n]* -> skip;
Block_comment : '/*' .*? '*/' -> skip;

// 790
fof_infix_unary             : fof_term Infix_inequality fof_term;
fof_atomic_formula          : fof_plain_atomic_formula
                            | fof_defined_atomic_formula
                            | fof_system_atomic_formula;
fof_plain_atomic_formula    : fof_plain_term;
fof_defined_atomic_formula  : fof_defined_plain_formula | fof_defined_infix_formula;
fof_defined_plain_formula   : fof_defined_term;
fof_defined_infix_formula   : fof_term defined_infix_pred fof_term;
fof_system_atomic_formula   : fof_system_term;

// 812
fof_plain_term           : constant
                        | functor '(' fof_arguments ')';
fof_defined_term        : defined_term | fof_defined_atomic_term;
fof_defined_atomic_term : fof_defined_plain_term;
fof_defined_plain_term  : defined_constant
                        | defined_functor '(' fof_arguments ')';
fof_system_term         : system_constant
                        | system_functor '(' fof_arguments ')';

// 839
fof_arguments           : fof_term (',' fof_term)*;
fof_term                : fof_function_term | variable;
// The original definition of fof_term includes additional components: tff_conditional_term | tff_let_term | tff_tuple_term
fof_function_term       : fof_plain_term | fof_defined_term
                        | fof_system_term;

// 890
// This is the root rule.
cnf_formula             : cnf_disjunction | '(' cnf_disjunction ')';
cnf_disjunction         : cnf_literal | cnf_disjunction Or cnf_literal;
cnf_literal             : fof_atomic_formula | Not fof_atomic_formula
                        | fof_infix_unary;

// 993
defined_infix_pred      : Infix_equality | Assignment;

// 999
constant                : functor;
functor                 : atomic_word;

// 1004
system_constant         : system_functor;
system_functor          : atomic_system_word;

// 1014
defined_constant        : defined_functor;
defined_functor         : atomic_defined_word;

// 1019
defined_term            : number | Distinct_object;
variable                : Upper_word;

// 1221
atomic_word : Lower_word | Single_quoted;
atomic_defined_word : Dollar_word;
atomic_system_word : Dollar_dollar_word;
number : Integer | Rational | Real;
