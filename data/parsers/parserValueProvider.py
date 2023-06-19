from typing import Dict
import locale

from data.dataset import GraphDataset
locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8') 

class ValueBackend:
    def get_nlu_val(self, bst: dict, var_name: str):
        raise NotImplementedError

    def get_table_val(self, table_name, func_name, values):
        raise NotImplementedError


class MockDB(ValueBackend):
    def get_nlu_val(self, var_name: str):
        if var_name == 'LAND':
            return "Deutschland"
        elif var_name == 'START':
            return "Berlin"
        return var_name + "VAR"

    def get_table_val(self, table_name, func_name, values):
        # print("TABLE NAME", table_name, "FUNC", func_name)
        # print("VALUES", values)
        if table_name == 'TAGEGELD':
           return 24
        else:
            return f"ERROR in Template: Tabelle {table_name} konnte nicht gefunden werden."


class RealValueBackend(ValueBackend):
    def __init__(self, a1_laender: Dict[str, bool], data: GraphDataset) -> None:
        self.a1_laender = a1_laender
        self.data = data

    def get_nlu_val(self, bst: dict, var_name: str):
        return bst[var_name]

    def get_table_val(self, table_name, func_name, values):
        if table_name == 'TAGEGELD':
            if func_name.upper() == "TAGEGELDSATZ":
                # FORM: TAGEGELD.tagegeldsatz(LAND, STADT)
                # TODO make this more generic? e.g. return list of possible cities for one country? or list of country that have this city
                land = values[0]
                stadt = values[1]
                # result = Tagegeld.objects.get(land=land, stadt=stadt).tagegeldsatz
                result = self.data.hotel_costs[land][stadt]
                return f"{result:g}"
            else:
                return f"ERROR in Template: In Tabelle {table_name} konnte Spalte {func_name} nicht gefunden werden."
        elif table_name == "A1LAENDER":
            if func_name.upper() == "BESCHEINIGUNG_NOTWENDIG":
                land = values[0]
                result = self.a1_laender[land]
                return bool(result)
            else:
                return f"ERROR in Template: In Tabelle {table_name} konnte Spalte {func_name} nicht gefunden werden."
        else:
            return f"ERROR in Template: Tabelle {table_name} konnte nicht gefunden werden."
