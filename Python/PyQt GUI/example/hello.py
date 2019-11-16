"""
------------------------------------------------
File Name: hello.py
Description:
Author: zhangtongxue
Date: 2019/10/26 8:59
-------------------------------------------------
"""

while True:
    print('请输入名字,输入Q退出程序：\n')

    name = input('输入名字：')

    if name == 'Q':
        break
    else:
        print('输入的名字是:', name)