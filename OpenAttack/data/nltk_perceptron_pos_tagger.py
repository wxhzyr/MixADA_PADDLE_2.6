"""
:type: function
:Size: 2.41MB

Model files for pos tagger in nltk.
`[code] <https://github.com/sloria/textblob-aptagger>`__
"""
from OpenAttack.utils import make_zip_downloader
import os

NAME = "TProcess.NLTKPerceptronPosTagger"

URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/averaged_perceptron_tagger.pickle.zip"
DOWNLOAD = make_zip_downloader(URL, "averaged_perceptron_tagger.pickle")


# def LOAD(path):
#     # ret = __import__("nltk").tag.PerceptronTagger(load=False)
#     # ret.load(os.path.join(path, "averaged_perceptron_tagger.pickle"))
#     import pickle
#     with open(os.path.join(path, "averaged_perceptron_tagger.pickle"), "rb") as f:
#         ret = pickle.load(f)
#     return ret.tag

# def LOAD(path):
#     import pickle
#     from nltk.tag.perceptron import PerceptronTagger

#     # 加载 pickle 文件
#     with open(os.path.join(path, "averaged_perceptron_tagger.pickle"), "rb") as f:
#         data = pickle.load(f)

#     # 如果加载的是 tuple，创建 PerceptronTagger 实例并加载数据
#     if isinstance(data, tuple):
#         tagger = PerceptronTagger(load=False)
#         tagger.model = data[0]  # 模型参数
#         tagger.tagdict = data[1]  # 词典
#         tagger.classes = data[2]  # 类别
#         return tagger.tag
#     else:
#         # 如果直接是 PerceptronTagger 对象
#         return data.tag

def LOAD(path):
    import pickle
    from nltk.tag.perceptron import PerceptronTagger, AveragedPerceptron

    # 加载 pickle 文件
    with open(os.path.join(path, "averaged_perceptron_tagger.pickle"), "rb") as f:
        data = pickle.load(f)

    # 如果加载的是 tuple，创建 PerceptronTagger 实例并加载数据
    if isinstance(data, tuple):
        tagger = PerceptronTagger(load=False)
        tagger.model = AveragedPerceptron()
        tagger.model.weights = data[0]  # 模型权重
        tagger.tagdict = data[1]  # 词典
        tagger.classes = set(data[2])  # 类别
        tagger.model.classes = tagger.classes
        return tagger.tag
    else:
        # 如果直接是 PerceptronTagger 对象
        return data.tag