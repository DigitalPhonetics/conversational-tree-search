from lark import Lark
from lark import Visitor


class AnswerTemplateParser:
    def __init__(self) -> None:
        self.grammar = """
            template: "{{" assignment "}}"

            ?assignment: NAME "=" TYPE -> var

            TYPE: "TEXT"
                | "NUMBER"
                | "BOOLEAN"
                | "TIMESPAN"
                | "TIMEPOINT"
                | "LOCATION"
            %import common.CNAME -> NAME
            %import common.WS_INLINE

            %ignore WS_INLINE
        """
        self.parser = Lark(self.grammar, start='template', parser='lalr')

    def find_variable(self, template: str):
        # parse & analyze template
        parse_tree = self.parser.parse(template)
        # print(parse_tree.pretty())
        
        finder = VariableFinder()
        finder.visit(parse_tree)
        return finder


class VariableFinder(Visitor):
    def __init__(self) -> None:
        super().__init__()
        self.name = None
        self.type = None

    def var(self, tree):
        self.name = tree.children[0].value
        self.type = tree.children[1].value
