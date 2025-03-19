import random
import re
import json

#把思考和回复分离出来的函数
def extract_think_response(text):
    span=re.search('朱璃在想的事：.+朱璃的回复',text,flags=re.S).span()
    think=text[span[0]:span[1]][8:-5]
    span=re.search('朱璃的回复.+',text,flags=re.S).span()
    res=text[span[0]:span[1]][7:]
    return think,res

#给思考加上<think>标记的函数
def process_think_response(think,response):
    return '<think>'+think+'</think>\n'+response


def process_user_prompt(aoikaze_dialog):
    content=''' 
你在风蓝动漫社的群聊中聊天，请根据多个群友的以下对话回复：[{dialog}]
'''
    content=content.replace('{dialog}',aoikaze_dialog)
    user_prompt={'role':'user','content':content}
    return user_prompt

#随机抽取一个systemprompt，作用是加强模型的鲁棒性
def generate_system_prompt():
    system_prompt_template=[
        '''你是朱璃，风蓝动漫社的看板娘之一。你的角色属性是"傲娇","宅女","小恶魔","雌小鬼","双马尾"。你有一个名为兰枫的姐姐。在风蓝动漫社中，你是知识渊博的交际花，偶尔喜欢开玩笑但是有时会很冷很怪，很在意自己在风蓝动漫社社员眼中美好的形象。有时潜水但时刻保持对风蓝各种群聊的观测，不会错过任何大事和乐子。你需要根据朱璃的设定回复动漫社的社员。
'''
    ]
    system_prompt=random.choice(system_prompt_template)
    return {'role':'system','content':system_prompt}

#读取文件
def dialog_load(filename):
    with open(filename,'r',encoding='utf-8') as f :
        a=f.readlines()
    dialoglist=[]
    for i in a:
        item=json.loads(i)
        dialoglist.append(item)
    return dialoglist

#将一个条目处理成数据集中的一个条目
def process_to_item(dialog):
    res=dialog['response']
    context=dialog['dialog']

    conversations=[]
    conversations.append(generate_system_prompt())
    conversations.append(process_user_prompt(context))
    think,response=extract_think_response(res)
    conversations.append({'role':'assistant','content':process_think_response(think,response)})    

    item={"conversations":conversations}
    return item


if __name__=='__main__':
    with open('../preprocess_data/dataset_1k.jsonl','a',encoding='utf-8') as f:
        dialoglist=dialog_load('../preprocess_data/process_dialog.json')
        for i in dialoglist:
            item=process_to_item(i)
            
            item_text=json.dumps(item,ensure_ascii=False)+'\n'
            f.write(item_text)