'''
基于脚本的任务型对话系统
'''
import json
import pandas as pd
import re

class DialogueSystem:
    def __init__(self):
        self.load()

    def load(self):
        self.nodes_info = {}
        self.load_scenario("./dailogue_system/scenario-买衣服.json")
        self.load_slot_template("./dailogue_system/slot_fitting_templet.xlsx")

    def load_scenario(self,scenario_file):
        #读取对话脚本
        with open(scenario_file,"r",encoding="utf-8") as f:
            self.scenario = json.load(f)
        scenario_name = scenario_file.split("/")[-1].split(".")[0]
        #保存node_info信息
        for node in self.scenario:
            self.nodes_info[scenario_name + node["id"]] = node
            if "childnode" in node:
                node["childnode"] = [scenario_name + child for child in node["childnode"]]

    def load_slot_template(self,slot_template_path):
        #读取槽位填充模板 slot query values
        self.slot_template = pd.read_excel(slot_template_path)
        self.slot_to_qv = {}
        for index,row in self.slot_template.iterrows():
            self.slot_to_qv[row["slot"]] = [row["query"],row["values"]]

    def nlu(self,memory):
        #意图识别
        memory = self.intent_recognition(memory)
        #槽位填充
        memory = self.slot_filling(memory)
        return memory

    def intent_recognition(self,memory):
        #意图识别：query和available_nodes中每个节点打分，选择分数最高的节点作为当前节点
        max_score = 0
        for node_name in memory["available_nodes"]:
            node_info = self.nodes_info[node_name]
            score = self.get_node_score(memory["query"],node_info)
            if score > max_score:
                max_score = score
                #保存命中的节点
                memory["hit_node"] = node_name
        return memory

    def get_node_score(self,query,node_info):
        score = 0
        intent_list = node_info["intent"]
        for intent in intent_list:
            score = max(score,self.sentence_match_score(query,intent))
        return score

    #计算两个字符串的相似度，使用jarcard距离
    def sentence_match_score(self,str1,str2):
        s1 = set(str1)
        s2 = set(str2)
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def slot_filling(self,memory):
        #槽位填充：根据当前命中节点的slot，对query进行槽位填充
        slot_list = self.nodes_info[memory["hit_node"]].get("slot",[])
        for slot in slot_list:
            slot_value = self.slot_to_qv[slot][1]
            if re.search(slot_value,memory["query"]):
                memory[slot] = re.search(slot_value,memory["query"]).group()
        return memory

    def dst(self,memory):
        #确认当前hit_node中所有的slot是否都已经被填充
        slot_list = self.nodes_info[memory["hit_node"]].get("slot",[])
        for slot in slot_list:
            if slot not in memory:
                memory["require_slot"] = slot
                return memory
        memory["require_slot"] = None
        #所有槽位填充完，移除该可用节点，开放子节点
        memory["available_nodes"] = self.nodes_info[memory["hit_node"]].get("childnode",[])
        return memory

    def dpo(self,memory):
        #如果有需要填充的slot，则选择一个query进行询问
        if memory["require_slot"] is not None:
            memory["policy"] = "ask"
        else:
            memory["policy"] = "reply"
        return memory

    def nlg(self,memory):
        #根据策略生成回复
        if memory["policy"] == "ask":
            memory["response"] = self.slot_to_qv[memory["require_slot"]][0]
        else:
            response = self.nodes_info[memory["hit_node"]]["response"]
            memory["response"] = self.slot_filling_response(response,memory)
        return memory
    
    def slot_filling_response(self,response,memory):
        #对回复进行槽位填充
        slot_list = self.nodes_info[memory["hit_node"]].get("slot",[])
        for slot in slot_list:
            if slot in response:
                response = response.replace(slot,memory[slot])
        return response

    def generate_response(self,query,memory):
        memory["query"] = query
        memory = self.nlu(memory) #自然语言理解 natural language understanding
        memory = self.dst(memory) #对话状态跟踪 dialogue state tracking
        memory = self.dpo(memory) #对话策略优化 dialogue policy optimization
        memory = self.nlg(memory) #自然语言生成 natural language generation
        return memory



if __name__ == '__main__':
    ds = DialogueSystem()
    # print(ds.nodes_info)
    # print("------------------------")
    # print(ds.slot_to_qv)

    #memory = {} #默认初始记忆
    #available_nodes = ['scenario-买衣服node1'] #初始默认可用的节点
    memory = {'available_nodes' : ['scenario-买衣服node1']}

    while True:
        query = input("User:")
        memory = ds.generate_response(query,memory)
        #print(memory)
        print('System:',memory["response"])
        if memory["available_nodes"] == []:
            break