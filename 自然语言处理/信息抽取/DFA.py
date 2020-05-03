import re
import copy


class Recognition:

    def __init__(self):
        # 读取敏感词表
        with open("./keywords", "r", encoding="utf-8") as file:
            self.sensitive_words = [line.replace("\n", "") for line in file.readlines()]

        # 构建 compiler
        self.sensitive_words_compiler = []
        for word in self.sensitive_words:
            try:
                self.sensitive_words_compiler.append(re.compile(word.replace(" ", "").replace("*", "\*")))
            except Exception as error:
                print(word)
    
    def find(self, content: str):
        result = []

        for word_compiler in self.sensitive_words_compiler:
            for item in word_compiler.finditer(content):
                result.append(item)
        
        return result


class DFA:

    def __init__(self, keyword_list: list):
        self.state_event_dict = self._generate_state_event_dict(keyword_list)

    def match(self, content: str):
        match_list = []
        state_list = []
        temp_match_list = []

        for char_pos, char in enumerate(content):
            if char in self.state_event_dict:
                state_list.append(self.state_event_dict)
                temp_match_list.append({
                    "start": char_pos,
                    "match": ""
                })

            for index, state in enumerate(state_list):
                is_find = False
                state_char = None

                if "*" in state:
                    state_list[index] = state["*"]
                    state_char = state["*"]
                    is_find = True

                if char in state:
                    state_list[index] = state[char]
                    state_char = state[char]
                    is_find = True

                if is_find:
                    temp_match_list[index]["match"] += char

                    if state_char["is_end"]:
                        match_list.append(copy.deepcopy(temp_match_list[index]))

                        if len(state_char.keys()) == 1:
                            state_list.pop(index)
                            temp_match_list.pop(index)
                else:
                    state_list.pop(index)
                    temp_match_list.pop(index)

        return match_list

    @staticmethod
    def _generate_state_event_dict(keyword_list: list) -> dict:
        state_event_dict = {}

        for keyword in keyword_list:
            current_dict = state_event_dict
            length = len(keyword)

            for index, char in enumerate(keyword):
                if char not in current_dict:
                    next_dict = {"is_end": False}
                    current_dict[char] = next_dict
                    current_dict = next_dict
                else:
                    next_dict = current_dict[char]
                    current_dict = next_dict

                if index == length - 1:
                    current_dict["is_end"] = True

        return state_event_dict


if __name__ == "__main__":

    with open("./keywords", "r", encoding="utf-8") as file:
        keyword_list = [line.replace("\n", "") for line in file.readlines()]

    # dfa = DFA(["匹配关键词", "匹配算法", "信息*取", "匹配"])
    # print(dfa.match("信息抽取之 DFA 算法匹配关键词，匹配算法，信息抓取"))
    dfa = DFA(keyword_list)
    print(dfa.match("法轮功不可取"))
