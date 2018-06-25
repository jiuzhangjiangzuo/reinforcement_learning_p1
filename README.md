# 九章算法 - 强化学习 - 实战项目1  

## 安装过程
1. 确认已经安装conda, 安装详情请参考之前的教程：https://www.jiuzhang.com/tutorial/ai-camp/477
2. 打开一个terminal，windows用户建议打开Anaconda Prompt.
3. 输入命令行: conda create --name rl_p1 python=3.5
4. 输入命令行: conda activate rl_p1   (有同学的环境激活命令是:source activate rl_p1,注意看环境创建完成之后，terminal上面的提示信息)。   
如果激活环境遇到了问题，conda会自己提示解决方案，比如：
```
CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
If your shell is Bash or a Bourne variant, enable conda for the current user with

    $ echo ". /Users/Andrew/anaconda3/etc/profile.d/conda.sh" >> ~/.bash_profile
```
NOTE： 注意这是我环境给出的解决方案(里面的Andrew是我的用户名)，不同的机器可能会给出不同的方案，要仔细看执行conda activate rl_p1之后, terminal里面给出的到底是什么内容。 

5. 输入命令行: git clone https://github.com/jiuzhangjiangzuo/reinforcement_learning_p1.git 或直接下载压缩包[link](https://github.com/jiuzhangjiangzuo/reinforcement_learning_p1/archive/master.zip)
6. 输入命令行: cd reinforcement_learning_p1
7. 输入命令行: pip install -r requirements.txt
8. 测试是否可以运行: python example.py
