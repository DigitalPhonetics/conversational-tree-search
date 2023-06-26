from lark import Lark
from lark import Transformer
from lark.visitors import  v_args

from data.parsers.parserValueProvider import MockDB, ValueBackend

class LogicTemplateParser:
    def __init__(self) -> None:
        self.grammar = """
            template: "{{" and "}}"

            ?and: or
                | and "AND" and -> _and
                | and "DEFAULT"
                | "(" and ")"
            
            ?or: statement 
                | and "OR" and    -> _or

            ?statement: sum "==" sum -> eq
                | sum "!=" sum      -> ne
                | sum ">" sum       -> gt
                | sum ">=" sum      -> ge
                | sum "<" sum       -> lt
                | sum "<=" sum      -> le
            
            ?sum: product
                | sum "+" product   -> add
                | sum "-" product   -> sub

            ?product: atom
                | product "*" atom  -> mul
                | product "/" atom  -> div

            ?atom: NUMBER          -> number
                | "TRUE"             -> trueval
                | "FALSE"            -> falseval
                | "-" atom         -> neg
                | NAME             -> var
                | const            -> const
                | NAME "." NAME "(" func_args? ")" -> func
                | "(" sum ")"
                

            ?func_args: (sum ",")* sum

            ?const: /"[^"]*"/

            %import common.CNAME -> NAME
            %import common.NUMBER
            %import common.WS_INLINE

            %ignore WS_INLINE
        """
        self.parser = Lark(self.grammar, start='template', parser='lalr')

    def parse_template(self, template: str, backend: ValueBackend, bst: dict):
        # parse & analyze template
        parse_tree = self.parser.parse(template)
        # print(parse_tree.pretty())

        # fill in constants, evaluate variables + functions
        return LogicTransformer(backend, bst).transform(parse_tree)

@v_args(inline=True)
class LogicTransformer(Transformer):
    from operator import add, sub, mul, truediv as div, neg, gt, ge, lt, le

    def __init__(self, backend: ValueBackend, bst: dict):
        self.backend = backend
        self.bst = bst

    def var(self, name):
        return self.backend.get_nlu_val(self.bst, name.value)

    def func(self, table_name, func_name, func_args):
        return self.backend.get_table_val(table_name.upper(), func_name, func_args)

    def func_args(self, *args):
        return args

    def eq(self, lhs, rhs):
        if isinstance(lhs, str): 
            # handle string comparison
            return lhs.lower().strip() == str(rhs).lower().strip()
        return lhs == rhs

    def ne(self, lhs, rhs):
        if isinstance(lhs, str):
            return lhs.lower().strip() != str(rhs).lower().strip()
        return lhs != rhs

    def number(self, num):
        return float(num)

    def const(self, content):
        return content.value.strip('"')

    def default(self, *args):
        return True
        
    def trueval(self):
        return True
    
    def falseval(self):
        return False

    def _or(self, *args):
        args = args[0]
        if len(args) == 1:
            return args[0]
        return args[0] or args[1]
    
    def _and(self, *args):
        args = args[0]
        if len(args) == 1:
            return args[0]
        return args[0] and args[1]

    def template(self, *args) -> bool:
        return args[0] # there is only 1 arg (final truth value)
