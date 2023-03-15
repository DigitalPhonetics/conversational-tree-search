from lark import Lark
from lark import Visitor
from lark import Transformer
from lark.visitors import v_args
from chatbot.adviser.app.parserValueProvider import MockDB, ValueBackend



class SystemTemplateParser:
    def __init__(self) -> None:
        self.grammar = """
            template: ("{{" sum "}}" | str )*
            ?sum: product
                | sum "+" product   -> add
                | sum "-" product   -> sub

            ?product: atom
                | product "*" atom  -> mul
                | product "/" atom  -> div

            ?atom: NUMBER          -> number
                | "-" atom         -> neg
                | NAME             -> var
                | const            -> const
                | NAME "." NAME "(" func_args? ")" -> func
                | "&nbsp;" atom
                | "(" sum ")"

            ?func_args: (sum ",")* sum

            ?str: inline_text+ -> text
            ?const: /"[^"]*"/
            ?inline_text:  /[^\{\}]+/ | "{" inline_text | "}" inline_text

            %import common.CNAME -> NAME
            %import common.NUMBER
            %import common.WS_INLINE

            %ignore WS_INLINE
        """
        self.parser = Lark(self.grammar, start='template', parser='lalr')

    def parse_template(self, template: str, backend: ValueBackend, bst: dict):
        # parse & analyze template
        parse_tree = self.parser.parse(template)

        # fill in constants, evaluate variables + functions
        return ValueTransformer(backend, bst).transform(parse_tree)

    def find_variables(self, template: str):
        # parse & analyze template
        parse_tree = self.parser.parse(template)
        # print(parse_tree.pretty())
        
        finder = VariableFinder()
        finder.visit(parse_tree)
        return finder.var_table


class VariableFinder(Visitor):
    def __init__(self) -> None:
        super().__init__()
        self.var_table = set()

    def var(self, tree):
        self.var_table.add(tree.children[0].value)

@v_args(inline=True)
class ValueTransformer(Transformer):
    from operator import add, sub, mul, truediv as div, neg

    def __init__(self, backend: ValueBackend, bst: dict):
        self.backend = backend
        self.bst = bst

    def var(self, name):
        return self.backend.get_nlu_val(self.bst, name.value)

    def func(self, table_name, func_name, func_args):
        return self.backend.get_table_val(table_name.upper(), func_name, func_args)

    def func_args(self, *args):
        return args

    def number(self, num):
        return float(num)

    def text(self, content):
        return content.value

    def const(self, content):
        return content.value

    def template(self, *args):
        return " ".join([str(arg).strip() for arg in args])


# if __name__ == "__main__":
#     # template_str = 'Test {{ myvar? }} and {{ myfunc( myfunc2(  myvar2?, "myconst",   "myconst2") ) }} as well as empty func {{func3()}}'
#     # template_str = 'Bei einer Abwesenheit von 24 Stunden am Kalendertag beträgt das Tagegeld in {{LAND}} {{TAGEGELD.satz(LAND, "24", "24")}} €.'
#     template_str = 'Bei einer {{ LAND }} Abwesenheit {{ (2 + 2) * 3 }} und {{ 3.0 * TAGEGELD.satz(2+3, "dasd", TAGEGELD.satz(STADT)) }}'
#     print("original template")
#     print(template_str)
#     print("===========")
#     parser = SystemTemplateParser()

#     variables = parser.find_variables(template_str)
#     print("Found varialbes:", variables)

#     print("Filled Template")
#     db = MockDB()
#     print(parser.parse_template(template_str, db))
