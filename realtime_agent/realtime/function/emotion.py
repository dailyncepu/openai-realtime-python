def get_respond_emotion(user_emotion,response_type):
    print("用户输入的情绪标签：", user_emotion)
    print("反馈用户使用的标签：", response_type)
    return "请以  " + response_type + " 的情绪回复用户输入。"



if __name__=="__main__":

    res = get_respond_emotion("开心","快乐")
    print(res)
