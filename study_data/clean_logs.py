import datetime
from typing import Tuple

def write_script(dialogs, out_file):
    with open(out_file, 'wt') as _file:
        for user_id in dialogs:
            _file.write(f"\n\nUSER ID: {user_id}\n")
            for line in dialogs[user_id]["dialog"]:
                _file.write(f"{line}")

    # with open("test.txt", "wt") as _file:
    #     for user_id in dialogs:
    #         for line in dialogs[user_id]["answer_candidates"]:
    #             if isinstance(line, Tuple):
    #                 _file.write(f"{line[0].strip()} ----- {line[1]}\n")

def create_script(in_file, dialogs={}):
    with open(in_file, 'rt') as _file:
        for line in _file:
            # TODO: switch to START-DIALOG and make sure to add user to dict HERE
            if "STARTED" in line:
                user_id = line.split()[-1].strip()
                if user_id not in dialogs:
                    dialogs[user_id] = {}
                    dialogs[user_id]["dialog"] = []
                    dialogs[user_id]["answer_candidates"] = []
                else:
                    dialogs[user_id]["dialog"].append("\tUSER: restart\n")
            elif "USR-UTTERANCE" in line:
                _, user_id, utterance = line.split(" # ")
                user_id = user_id[5:].strip()
                speaker = "USER"
                utterance = utterance.replace('USR-UTTERANCE (reisekosten) -', '')
                print(1, utterance)
                unique = utterance.split("-")[-1].strip() == "True"
                print(2, utterance)
                if unique and dialogs[user_id]["answer_candidates"]:
                    # print(dialogs[user_id]["answer_candidates"][-1])
                    dialogs[user_id]["answer_candidates"][-1] = (utterance, dialogs[user_id]["answer_candidates"][-1])
                if user_id not in dialogs:
                    print(user_id, "UTTERANCE")
                    dialogs[user_id] = {}
                    dialogs[user_id]["dialog"] = []
                print(3, utterance)
                dialogs[user_id]["dialog"].append(f"{speaker}: {utterance}")
            elif "POLICY" in line:
                speaker = "SYSTEM"
                _, turn_info = line.split(" - ")
                turn_info = turn_info.split(", ")
                end_char = turn_info[0].find(":")
                user_id = turn_info[0][7:end_char].strip()
                utterance = ", ".join(turn_info[3:])[6:]
                if len(utterance.split("'")) > 1:
                    utterance = utterance.split("'")[1]
                    utterance = utterance.replace("</p>", "").replace("<p>", "").replace("<strong>", "").replace("</strong>", "").replace("<br />", "")
                    utterance = utterance.replace("&uuml;", "ü").replace("&auml;", "ä").replace("&ouml;", "ö").replace("&szlig;", "ß").replace("&Uuml;", "Ü").replace("&Auml;", "Ä").replace("&Ouml;", "Ö")
                    utterance = utterance.replace("&euro;", "€").replace("\\n<ul>\\n<li>", " *").replace("</li>", "").replace("\\n<li>", "*").replace("&nbsp;", " ").replace("\\n</ul>", "").replace("\\n", " ")
                    utterance += "\n"
                if user_id not in dialogs:
                    print(user_id, "NLG")
                    dialogs[user_id] = {}
                    dialogs[user_id]["dialog"] = []
                dialogs[user_id]["dialog"].append(f"{speaker}: {utterance}")
            elif "ANSWER-CANDIDATES" in line:
                _, user_id, candidates = line.split(" - ")
                candidates = candidates.strip()
                user_id = user_id.split(" # ")[1][5:].strip()
                if user_id in dialogs:
                    dialogs[user_id]["answer_candidates"].append(candidates)
                else:
                    print(user_id)

if __name__ == '__main__':
    dialogs = {}
    create_script("log_2022-03-08_15-27-13.log", dialogs=dialogs)
    write_script(dialogs, "test_log.txt")

